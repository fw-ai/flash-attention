"""Focused parity for the opt-in hd256 redundant-causal-mask pruning path."""

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.testing import attention_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="dedicated hd256 2CTA kernel requires SM100-family CUDA",
)

_HD256_PRUNE_COMPILE_KEY_INDEX = 14


def _set_candidate(enabled: bool) -> None:
    from flash_attn.cute import utils

    utils._fa_hd256_prune_redundant_causal_masking = enabled


def _run_backward(q, k, v, out, dout, lse, *, causal: bool, candidate: bool, **kwargs):
    _set_candidate(candidate)
    return _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q.shape[-1] ** -0.5,
        causal,
        0.0,
        deterministic=False,
        dq=torch.empty_like(q),
        dk=torch.empty_like(k),
        dv=torch.empty_like(v),
        **kwargs,
    )


def _assert_exact(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    assert torch.count_nonzero(reference != candidate).item() == 0
    assert torch.isfinite(candidate).all()


def _is_opt_in_bwd_key(key) -> bool:
    return (
        len(key) > _HD256_PRUNE_COMPILE_KEY_INDEX
        and key[0] // 10 in (10, 11)
        and key[_HD256_PRUNE_COMPILE_KEY_INDEX] is True
    )


def _run_guarded_noop_candidate(*args, **kwargs):
    """Run env-on while proving the full backward key set is unchanged."""
    cache = _flash_attn_bwd.compile_cache.cache
    saved_opt_in = {
        key: compiled for key, compiled in cache.items() if _is_opt_in_bwd_key(key)
    }
    for key in saved_opt_in:
        del cache[key]
    keys_before = set(cache)
    keys_after = set()
    try:
        result = _run_backward(*args, candidate=True, **kwargs)
        keys_after = set(cache)
    finally:
        for key in set(cache) - keys_before:
            del cache[key]
        cache.update(saved_opt_in)
    assert keys_after == keys_before
    return result


def _run_expected_opt_in_candidate(*args, **kwargs):
    """Run the proven shape while requiring exactly one opt-in backward key."""
    cache = _flash_attn_bwd.compile_cache.cache
    saved_opt_in = {
        key: compiled for key, compiled in cache.items() if _is_opt_in_bwd_key(key)
    }
    for key in saved_opt_in:
        del cache[key]
    keys_before = set(cache)
    added_keys = set()
    try:
        result = _run_backward(*args, candidate=True, **kwargs)
        added_keys = set(cache) - keys_before
    finally:
        cache.update(saved_opt_in)
    assert len(added_keys) == 1
    assert all(_is_opt_in_bwd_key(key) for key in added_keys)
    return result


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [128, 129, 255, 256, 257, 383, 384, 385])
def test_hd256_causal_mask_prune_reference_parity_and_determinism(seqlen, causal):
    torch.manual_seed(17_000 + seqlen + int(causal))
    q_ref = torch.randn(
        1, seqlen, 4, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    k_ref = torch.randn(
        1, seqlen, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    v_ref = torch.randn(
        1, seqlen, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    q, k, v = [tensor.detach() for tensor in (q_ref, k_ref, v_ref)]

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    try:
        _set_candidate(False)
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=256**-0.5,
            causal=causal,
            return_lse=True,
        )
        dout = torch.randn_like(out)
        baseline = _run_backward(
            q, k, v, out, dout, lse, causal=causal, candidate=False
        )
        candidate = (
            _run_expected_opt_in_candidate(q, k, v, out, dout, lse, causal=True)
            if causal and seqlen == 128
            else _run_backward(q, k, v, out, dout, lse, causal=True, candidate=True)
            if causal
            else _run_guarded_noop_candidate(q, k, v, out, dout, lse, causal=False)
        )
        candidate_repeat = _run_backward(
            q, k, v, out, dout, lse, causal=causal, candidate=True
        )
    finally:
        _set_candidate(False)

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item() + fwd_atol

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), dout)
    for produced, expected in zip(candidate, baseline):
        _assert_exact(expected, produced)
    for produced, repeated, reference, pytorch_reordered in zip(
        candidate,
        candidate_repeat,
        (dq_ref, dk_ref, dv_ref),
        (dq_pt, dk_pt, dv_pt),
    ):
        _assert_exact(produced, repeated)
        if causal:
            atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
            assert (produced - reference).abs().max().item() <= 2 * (
                pytorch_reordered - reference
            ).abs().max().item() + atol


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k", [(257, 255), (255, 257), (386, 384), (384, 386)]
)
def test_hd256_causal_mask_prune_unequal_qk_is_guarded_noop(seqlen_q, seqlen_k):
    torch.manual_seed(29_000 + seqlen_q + seqlen_k)
    q_ref = torch.randn(
        1, seqlen_q, 4, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    k_ref = torch.randn(
        1, seqlen_k, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    v_ref = torch.randn(
        1, seqlen_k, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    q, k, v = [tensor.detach() for tensor in (q_ref, k_ref, v_ref)]

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=True)
    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=True,
        upcast=False,
        reorder_ops=True,
    )

    try:
        _set_candidate(False)
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=256**-0.5,
            causal=True,
            return_lse=True,
        )
        dout = torch.randn_like(out)
        baseline = _run_backward(q, k, v, out, dout, lse, causal=True, candidate=False)
        candidate = _run_guarded_noop_candidate(q, k, v, out, dout, lse, causal=True)
        candidate_repeat = _run_backward(
            q, k, v, out, dout, lse, causal=True, candidate=True
        )
    finally:
        _set_candidate(False)

    for produced, expected in zip(candidate, baseline):
        _assert_exact(expected, produced)
    for produced, repeated in zip(candidate, candidate_repeat):
        _assert_exact(produced, repeated)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), dout)
    for produced, reference, pytorch_reordered in zip(
        candidate,
        (dq_ref, dk_ref, dv_ref),
        (dq_pt, dk_pt, dv_pt),
    ):
        atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
        assert (produced - reference).abs().max().item() <= 2 * (
            pytorch_reordered - reference
        ).abs().max().item() + atol


def test_hd256_causal_mask_prune_varlen_is_guarded_noop():
    seqlen = 257
    torch.manual_seed(41_257)
    q_ref = torch.randn(
        1, seqlen, 4, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    k_ref = torch.randn(
        1, seqlen, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    v_ref = torch.randn(
        1, seqlen, 1, 256, device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    q, k, v = [
        tensor.detach().squeeze(0).contiguous() for tensor in (q_ref, k_ref, v_ref)
    ]
    cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
    varlen_kwargs = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": seqlen,
        "max_seqlen_k": seqlen,
    }

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=True)
    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=True,
        upcast=False,
        reorder_ops=True,
    )

    try:
        _set_candidate(False)
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=256**-0.5,
            causal=True,
            return_lse=True,
            **varlen_kwargs,
        )
        dout = torch.randn_like(out)
        baseline = _run_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            causal=True,
            candidate=False,
            **varlen_kwargs,
        )
        candidate = _run_guarded_noop_candidate(
            q,
            k,
            v,
            out,
            dout,
            lse,
            causal=True,
            **varlen_kwargs,
        )
        candidate_repeat = _run_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            causal=True,
            candidate=True,
            **varlen_kwargs,
        )
    finally:
        _set_candidate(False)

    for produced, expected in zip(candidate, baseline):
        _assert_exact(expected, produced)
    for produced, repeated in zip(candidate, candidate_repeat):
        _assert_exact(produced, repeated)

    dout_ref = dout.unsqueeze(0)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout_ref
    )
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), dout_ref)
    for produced, reference, pytorch_reordered in zip(
        candidate,
        (dq_ref.squeeze(0), dk_ref.squeeze(0), dv_ref.squeeze(0)),
        (dq_pt.squeeze(0), dk_pt.squeeze(0), dv_pt.squeeze(0)),
    ):
        atol = 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
        assert (produced - reference).abs().max().item() <= 2 * (
            pytorch_reordered - reference
        ).abs().max().item() + atol
