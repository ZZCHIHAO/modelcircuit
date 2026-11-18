# S1 core implementation and evidence

This is the implementer's phase record. The final integrated acceptance on 2026-11-14 is in [../VERIFICATION.md](../VERIFICATION.md): 23 core tests within 61 backend tests passed. Use that record and `frontend/design-qa.md` for current acceptance and scope limits.

Owned implementation files: `studio_core/__init__.py`, `schema.py`, `examples.py`, `runtime.py`, and `tests/test_core.py`.

The validator is standard-library-only. It validates tensor sources, exact named arguments, parameter values, shape compatibility, dependency order, recursive definitions, chip bindings, and identifiers that conflict with PyTorch module storage. It resolves each chip instance into a private graph copy, so public parameter overrides never mutate a shared definition. Parameter counts count each module instance once, even when called on two branches. Nested chips bind through their public parameters.

The runtime constructs a CPU `ModuleDict` per graph scope. Repeated calls look up the same component instance, with separate per-call tensor values. Root forward returns all named outputs. The experiment uses a fixed seeded synthetic batch; labels are the sign of the sum of each input's feature mean. Loss before, loss after, accuracy, and probes are measured on that same batch in evaluation mode. They do not estimate held-out performance. History contains pre-update training loss. Gradient norms come from the last optimization step; update norms compare final parameters with their initial values. Probe statistics cover the full tensor; values are sampled to at most 256 scalars, with an explicit sample size. Frozen gradients remain unavailable and frozen updates remain zero.

The example root ID is `shared_classifier`: the contract's illustrative `shared-classifier` conflicts with its own identifier regex. All example view IDs and semantics follow the contract.

## Resource boundaries

Validation bounds the raw document before deepcopy: 20,000 JSON values, nesting depth 32, strings 4,096 characters, at most 32 definitions and 512 entries in each graph collection. Chip nesting is limited to six levels; resolved chip copies have a cumulative 2,048-node budget. Shape inference has a 16,384-node work budget. Models have at most 2,000,000 unique parameters, rank-eight tensors, and 1,000,000 elements per inferred tensor. Unknown batch dimensions are estimated at the maximum supported batch size 64. Cumulative inferred tensor elements are bounded at 16,000,000. Runtime additionally checks concrete primitive output sizes before allocation and cumulative observed tensor sizes, covering dynamic batch expansion through Concat. These are rejection limits, not automatic model resizing. HTTP request, concurrency, disk, and process time limits belong to the coordinator.

## Verification

Executed with `studio/backend` in `PYTHONPATH`:

```text
studio/.venv/Scripts/python.exe -m unittest discover -s studio/backend/tests -p test_core.py -v
```

Initial execution before implementation produced assertion failures for missing core modules. After the parent installed PyTorch 2.6.0+cpu, the complete core suite passed: 18 tests, no skips, no failures, 2.773 seconds. An initial resource-boundary run exposed seven failures before budgets were implemented. Additional red-to-green cases covered allocation after dynamic Concat expansion and overflow exceptions from extreme integer Dropout, learning-rate, and position values. The final suite verifies rejection before weight allocation for oversized models, malformed-field coverage with 90 deterministic substitutions, recursive and exponentially expanded chips, exact shapes and counts, and module attribute conflicts.

Real runtime tests compare shared forward outputs, backward gradients, and one SGD update with an independent handwritten PyTorch reference. They also verify nested bound dimensions, named outputs, reproducibility, actual probe statistics and bounded samples, frozen gradients and updates, cancellation/configuration errors, disconnected trainable paths, and unbatched concatenation without implicit Add broadcasting.

Default experiment measured on CPU with seed 42, 20 steps, learning rate 0.05, and batch size 16: loss 0.9259226322174072 before training, 0.49365317821502686 after training, same-batch accuracy 0.9375, 58 unique parameters, and the shared encoder called by `encode_a` and `encode_b`. These are synthetic training observations.

The code-bridge suite was rerun after the runtime resource checks: all 19 tests passed in 5.138 seconds, including standalone exported-model execution and nested shared output/gradient agreement. An earlier combined discovery had nine coordinator failures because `server.py` was still being implemented by the parent; no core or bridge test failed.

```text
studio/.venv/Scripts/python.exe -m compileall -q studio/backend/studio_core studio/backend/tests/test_core.py
```

This syntax check passed. No GUI dependency, original ML Forge business-module modification, Git initialization, or dependency installation was performed by the core agent.
