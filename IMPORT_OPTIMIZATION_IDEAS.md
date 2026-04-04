# Fire Attention: Python Import Optimization Analysis & Ideas

## Problem Statement

Importing `torch`, `flash_attn`, and the broader Fireworks stack takes ~6 seconds per process. With multiprocessing workers, this cost is paid multiple times (main process + each worker), adding ~12+ seconds of cold-start latency. Python's GIL and CPython import machinery make it difficult to parallelize these imports effectively (observed: 16 workers import in ~12s instead of the expected ~7s of a single process, due to contention).

---

## Deep Dive: What Actually Gets Imported

### The core `flash_attn` import chain

```
import flash_attn
  └── flash_attn.__init__
        └── flash_attn.flash_attn_interface
              ├── torch (+ torch.nn)           ← ~3-4s alone
              └── flash_attn_2_cuda            ← CUDA extension .so load
```

### The model-loading fan-out (`gpt.py`)

When any model code is touched (e.g., `from flash_attn.models.gpt import GPTLMHeadModel`), the import graph explodes:

```
flash_attn.models.gpt
  ├── torch, torch.nn, torch.nn.functional
  ├── einops
  ├── transformers.GPT2Config              ← pulls in all of HuggingFace transformers
  ├── flash_attn.models.bigcode            ← imports transformers.GPT2Config, GPTBigCodeConfig
  ├── flash_attn.models.falcon             ← imports transformers.FalconConfig, GPT2Config
  ├── flash_attn.models.gpt_neox           ← imports transformers.GPT2Config, GPTNeoXConfig
  ├── flash_attn.models.gptj               ← imports transformers.GPT2Config, GPTJConfig
  ├── flash_attn.models.llama              ← imports transformers.GPT2Config, LlamaConfig, sentencepiece
  ├── flash_attn.models.opt                ← imports transformers.GPT2Config, OPTConfig
  ├── flash_attn.modules.block             ← imports torchvision.ops.StochasticDepth (!)
  │     ├── flash_attn.modules.mha         ← imports flash_attn (CUDA ext), rotary, fused_dense
  │     └── flash_attn.modules.mlp         ← imports fused_dense, activations
  ├── flash_attn.modules.embedding
  ├── flash_attn.ops.activations
  ├── flash_attn.ops.fused_dense           ← imports fused_dense_lib CUDA extension
  ├── flash_attn.ops.triton.mlp            ← imports triton
  ├── flash_attn.ops.triton.layer_norm     ← imports triton
  ├── flash_attn.utils.distributed
  ├── flash_attn.utils.generation          ← imports transformers.generation
  └── flash_attn.utils.pretrained          ← imports transformers.utils, safetensors
```

### The heavyweight third-party imports involved

| Import | Approx. cost | Where triggered |
|--------|-------------|-----------------|
| `torch` | ~3-4s | Everywhere (unavoidable for core) |
| `transformers` | ~1-2s | `gpt.py`, `pretrained.py`, `generation.py`, all model files |
| `torchvision` | ~0.5-1s | `block.py` via `StochasticDepth` |
| `einops` | ~0.1s | Many files |
| `triton` | ~0.5s | `ops/triton/` modules |
| `safetensors` | ~0.1s | `pretrained.py` |
| `sentencepiece` | ~0.1s | `llama.py` |
| CUDA extension `.so` loads | ~0.1-0.3s each | `flash_attn_2_cuda`, `fused_dense_lib`, `dropout_layer_norm` |

---

## Optimization Ideas

### TIER 1: High Impact, Moderate Effort — Lazy Import Restructuring

#### Idea 1: Make `flash_attn/__init__.py` lazy via PEP 562

**Current state:** `flash_attn/__init__.py` eagerly imports 7 functions from `flash_attn_interface`, which triggers `import torch` + loading the CUDA `.so`.

**Proposed:** Use `__getattr__` (PEP 562) to defer all imports until first attribute access:

```python
__version__ = "2.5.6"

def __getattr__(name):
    if name in _PUBLIC_API:
        from flash_attn.flash_attn_interface import ...
        return globals()[name]
    raise AttributeError(...)
```

**Impact:** `import flash_attn` becomes nearly instant (~milliseconds). Cost deferred to first actual function call. This alone won't help if downstream code immediately uses the functions, but it helps if `flash_attn` is imported transitively by code that doesn't use it.

#### Idea 2: Defer all `transformers` imports to function scope

**Current state:** 12 files under `flash_attn/` import from `transformers` at module level. These are only needed when:
- Loading pretrained models (`from_pretrained`)
- Using HF generation utilities
- Converting state dicts between formats

**Proposed:** Move all `transformers` imports inside the functions that use them:

```python
# In gpt.py — move GPT2Config import to where it's actually used
# In pretrained.py — move transformers.utils imports inside state_dict_from_pretrained()
# In generation.py — move transformers.generation imports inside decode functions
```

**Impact:** Saves ~1-2s per process. Transformers is one of the heaviest imports after torch itself, and it's only needed for model loading/conversion, not inference.

#### Idea 3: Defer model remapper imports in `gpt.py`

**Current state:** `gpt.py` eagerly imports remapping functions from ALL model types (llama, falcon, opt, bigcode, gptj, gpt_neox) at module level. Each of those files eagerly imports their own `transformers.*Config`.

**Proposed:** Move these imports inside `from_pretrained()` where they're actually used, behind the `if model_name.startswith(...)` branches:

```python
@classmethod
def from_pretrained(cls, model_name, ...):
    ...
    if model_name.startswith("meta-llama/Llama-"):
        from flash_attn.models.llama import remap_state_dict_hf_llama
        state_dict = remap_state_dict_hf_llama(state_dict, config)
    ...
```

**Impact:** Eliminates importing 6 model files + their transitive `transformers` imports during normal model construction. Only the model type actually being loaded gets imported.

#### Idea 4: Remove `torchvision` dependency from `block.py`

**Current state:** `block.py` has `from torchvision.ops import StochasticDepth`. This pulls in the entire torchvision package for a single trivial class.

**Proposed:** Inline a minimal `StochasticDepth` implementation (it's ~15 lines: random dropout of entire residual paths). Or defer the import to `Block.__init__` only when `drop_path > 0`.

**Impact:** Eliminates ~0.5-1s torchvision import for ALL code paths that use `Block` (which is essentially all model code).

#### Idea 5: Defer CUDA extension loading in `flash_attn_interface.py`

**Current state:** Line 10: `import flash_attn_2_cuda as flash_attn_cuda` — loaded at module import time.

**Proposed:** Load the CUDA extension lazily on first use:

```python
_flash_attn_cuda = None

def _get_cuda():
    global _flash_attn_cuda
    if _flash_attn_cuda is None:
        import flash_attn_2_cuda
        _flash_attn_cuda = flash_attn_2_cuda
    return _flash_attn_cuda
```

**Impact:** ~0.1-0.3s saved. More importantly, allows importing the Python layer without requiring CUDA, which enables better fork/forkserver strategies.

---

### TIER 2: High Impact, Higher Effort — Process-Level Optimizations

#### Idea 6: Pre-import warm pool with deferred CUDA init

**Concept:** The fundamental issue with `forkserver` is that CUDA initialization + fork = broken. But what if we import everything EXCEPT CUDA-touching code before the fork point?

**Implementation:**
1. At startup, create a "template" process that imports `torch` (without CUDA init), `transformers`, `einops`, etc.
2. Use this as the forkserver template
3. After forking, each worker initializes CUDA independently

**Key technical requirement:** Must ensure `import torch` does NOT trigger CUDA initialization. This means:
- Set `CUDA_VISIBLE_DEVICES=""` during the import phase
- Or use `PYTORCH_NVML_BASED_CUDA_CHECK=1` (avoids CUDA driver init during torch import)
- Then restore CUDA visibility after fork

**Challenge:** Some flash_attn code (CUDA extensions) triggers CUDA init during import. This is why Idea 5 (deferred CUDA ext loading) is a prerequisite.

**Impact:** Could reduce per-worker import cost from ~6s to ~0.5s (only CUDA init + extension loading).

#### Idea 7: Interpreter state snapshot and restore

**Concept:** Use Python's `marshal` module or a custom mechanism to snapshot `sys.modules` after full import, serialize it, and restore it in new worker processes instead of re-importing.

**Approaches:**
- **`fork()` with CUDA workaround:** The cleanest solution if CUDA init can be deferred
- **`torch.package`:** Package all needed modules into a zip that can be loaded faster
- **Custom `importlib` loader:** That loads pre-compiled `.pyc` from a shared tmpfs/mmap

**Impact:** Potentially massive — worker startup could be <1s.

#### Idea 8: Persistent worker daemon with IPC

**Concept:** Instead of spawning new worker processes for each request, maintain a pool of long-lived workers that have already completed imports. Workers receive work items via IPC (shared memory / unix sockets / pipes).

**Implementation:**
- At system startup, spawn N worker processes that do all imports
- Workers sit in an event loop waiting for work
- Main process sends serialized task descriptions via Queue/Pipe
- Workers process and return results

**The Queue passing issue from the Slack thread:** The challenge of "passing Queue down to the processes" could be addressed by:
- Creating Queues at the top level and passing them via process constructors (not as function args)
- Using `multiprocessing.Manager` for shared state
- Using raw `os.pipe()` + file descriptors inherited across fork
- Using Unix domain sockets for communication

**Impact:** Eliminates import cost for steady-state (one-time cost at system startup). The Slack thread notes "mysterious contention" with 16 workers — this likely isn't import contention but rather filesystem/IO contention from Python's import machinery hitting the same `.py`/`.pyc` files simultaneously.

---

### TIER 3: Systemic / Infrastructure-Level

#### Idea 9: Bytecode pre-compilation and filesystem optimization

**Concept:** Python's import machinery does a lot of `stat()` and `open()` calls. With many packages, this can be slow, especially if the filesystem has high latency.

**Actions:**
- Pre-compile all `.pyc` files in the container image: `python -m compileall -b .`
- Use `PYTHONDONTWRITEBYTECODE=1` at runtime (skip .pyc generation attempts)
- Mount python site-packages from a ramdisk / tmpfs
- Use `zipimport` (put site-packages in a `.zip` file — single file seek vs. thousands)

**Impact:** ~10-20% reduction in import time due to fewer filesystem operations.

#### Idea 10: Import profiling instrumentation

**Concept:** Before optimizing, measure. Add import-time profiling to identify the actual biggest contributors.

**Actions:**
- `python -X importtime -c "import flash_attn"` — built-in import timing
- Custom import hook that logs wall-clock time per module
- Profile with `py-spy` or `perf` during startup

**This should be done first** to validate assumptions about where time is spent.

#### Idea 11: Vendoring minimal transformers subset

**Concept:** If only `GPT2Config`, `LlamaConfig`, `FalconConfig`, etc. are needed, vendor just those classes instead of pulling in the entire `transformers` package.

**Implementation:** Copy the ~10 config classes needed from transformers into `flash_attn/configs/`. These are pure dataclasses with no complex dependencies.

**Impact:** Eliminates the `transformers` import entirely if combined with Idea 2. Saves ~1-2s.

**Risk:** Maintenance burden of keeping vendored code in sync.

#### Idea 12: Address parallel import contention

**The mystery from the Slack thread:** "One process is 7 seconds, with a pool of 16 we get 12 seconds" — imports should be independent but they contend.

**Root causes to investigate:**
1. **Python's import lock:** CPython has a global import lock. Even in separate processes, if they're competing for shared filesystem resources (NFS, overlayfs), there can be contention.
2. **Filesystem I/O:** 16 processes simultaneously stat/reading thousands of .py/.pyc files thrashes the page cache and filesystem.
3. **CUDA driver initialization:** All processes trying to initialize the CUDA driver simultaneously — CUDA has internal locks.
4. **`.pyc` file generation:** If `.pyc` files don't exist, 16 processes simultaneously try to generate them (file locking contention).
5. **`dlopen()` contention:** Loading `.so` files (CUDA extensions, torch internals) involves `dlopen` which takes a process-wide lock in glibc.

**Mitigations:**
- Pre-compile all `.pyc` files (Idea 9)
- Stagger worker starts: start workers with 200ms delays between them
- Use `PYTHONDONTWRITEBYTECODE=1` to avoid pyc write contention
- Pre-populate the page cache by importing once before spawning workers
- Use `fork` instead of `spawn` for workers (shares already-loaded modules in memory)

#### Idea 13: Two-phase module loading architecture

**Concept:** Redesign the import structure into two phases:
1. **Phase 1 (pre-fork):** Import everything that doesn't touch CUDA: torch core (careful!), transformers configs, einops, pure Python modules
2. **Phase 2 (post-fork per-worker):** Load CUDA extensions, initialize CUDA context

**Implementation:**
```python
# Phase 1: Safe to share via fork
import torch  # with CUDA_VISIBLE_DEVICES="" or PYTORCH_NVML_BASED_CUDA_CHECK=1  
import transformers
import einops
# ... all pure Python imports

# Fork workers here (forkserver or regular fork)

# Phase 2: Each worker independently  
import flash_attn_2_cuda
torch.cuda.init()
```

**Key challenge:** Ensuring `import torch` doesn't call into CUDA. PyTorch lazily initializes CUDA, but some codepaths trigger it earlier (e.g., `torch.cuda.is_available()` may call into the driver).

#### Idea 14: Use `multiprocessing.shared_memory` for module bytecode

**Concept:** Load compiled Python bytecode into shared memory that workers can map directly, avoiding per-process filesystem reads.

**This is exotic** and would require a custom import loader, but it would eliminate the filesystem I/O bottleneck for parallel imports.

---

### TIER 4: Radical / Long-term Ideas

#### Idea 15: PyPy or alternative Python runtimes

PyPy has a JIT that makes imports faster after warmup. However, compatibility with torch CUDA extensions is poor.

#### Idea 16: Nuitka/Cython compilation of the Python layer

Compile the pure-Python parts of flash_attn and key dependencies into C extensions that load faster than interpreted Python.

#### Idea 17: Static analysis + tree-shaking for imports

Build a tool that analyzes which functions are actually called in a given deployment and generates a minimal import set, removing dead code paths at "compile time."

#### Idea 18: `torch.deploy` / Frozen modules

Use `torch.deploy` (multi-interpreter embedding) or Python's frozen module mechanism to embed the needed modules directly into the binary, avoiding filesystem-based imports entirely.

---

## Recommended Prioritization

| Priority | Idea | Effort | Expected Impact |
|----------|------|--------|-----------------|
| **P0** | **#10: Import profiling** | Low | Foundation for all other work |
| **P0** | **#2: Defer `transformers` imports** | Low | ~1-2s per process |
| **P0** | **#3: Defer model remappers in `gpt.py`** | Low | ~0.5-1s per process |
| **P0** | **#4: Remove torchvision from `block.py`** | Low | ~0.5-1s per process |
| **P1** | **#1: Lazy `__init__.py`** | Low | Enables faster transitive imports |
| **P1** | **#5: Defer CUDA extension loading** | Medium | ~0.3s + enables fork strategies |
| **P1** | **#12: Fix parallel import contention** | Medium | Key for multi-worker case |
| **P1** | **#9: Bytecode pre-compilation** | Low | ~10-20% speedup |
| **P2** | **#6: Pre-import warm pool** | High | Could reduce worker cost to <1s |
| **P2** | **#8: Persistent worker daemon** | High | Eliminates per-request import cost |
| **P2** | **#13: Two-phase loading** | High | Fundamental architecture improvement |
| **P3** | **#11: Vendor transformers configs** | Medium | Eliminates transformers entirely |
| **P3** | **#7: Interpreter state snapshot** | Very High | Radical but potentially game-changing |

## Quick Wins Summary

The P0 items are **pure code changes within this repo** that require no architectural changes:
1. Move `from transformers import ...` from module-level to function-level in ~12 files
2. Move model remapper imports into the `if` branches of `from_pretrained()`
3. Replace `from torchvision.ops import StochasticDepth` with a local 15-line implementation
4. Add `python -X importtime` profiling to validate the impact

These alone could save **2-4 seconds per process**, cutting the 6s import time roughly in half.
