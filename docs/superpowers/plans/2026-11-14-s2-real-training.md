# S2 Real Training Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Ownership and exact interfaces are in studio/S2_CONTRACT.md; read it before editing. Test missing behavior, observe failure, implement, then independently review.

**Goal:** Deliver native real CSV/2D image classification with train/val/test, reproducible artifacts/checkpoints, CPU/single-CUDA selection and a tested EXE; report GPU verification separately from CPU availability.

**Architecture:** Preserve the independent graph core, add dataset/task/device profiles, extend the local worker coordinator and compose native experiment/workspace panels. Keep S1 as a regression mode. Multi-GPU orchestration remains a later milestone.

**Spec:** `studio/S2_CONTRACT.md`.

## Constraints and rulings

- No Use Computer or OS/browser input automation for debugging. Use unit/integration tests, application-owned DPG rendering and frozen diagnostics. GitHub upload remains deferred.
- User requested continued implementation after identifying real-data/GPU gaps; no repeated approval gate for routine changes. The user approved an independent CUDA runtime; torch2.6.0+cu124 was installed without replacing CPU, and source/frozen CSV/CNN GPU training and native callbacks passed.
- One RTX4060 Laptop8GB was detected; current torch is2.6.0+cpu and GPU was95% utilized at inspection. Do not claim CUDA or multi-GPU training verified from nvidia-smi alone.
- Existing Git has an unborn main and no configured user/remote; preserve its staged baseline and S1 source/portable archives. Do not invent commits or create a worktree from a nonexistent HEAD. No public publishing.
- No clinical dataset/model downloads or unrequested full experiments. Local generated CSV/PNG fixtures exercise file-based pipelines and are explicitly test fixtures.

## Tasks

- [x] Model/CNN and code bridge: add four primitives, strict shared shape rules, per-profile budgets, device/capture-aware execution, native defaults; S1 tests plus handwritten forward/gradient/update and CNN code-roundtrip tests.
- [x] Dataset/training: deterministic manifests, group/split leakage checks, train-only scaling, loaders, epoch training, val-only model selection, held-out test, checkpoints and exact CPU resume; file-based CSV/PNG fixtures and bad/mutated-source/cancellation tests.
- [x] Root config/device/coordinator: finite whitelisted ExperimentSpec, capability query/resolve, artifact directories, per-run limits, exclusive CUDA reservation, worker dispatch and lifecycle tests.
- [x] Native workspace/panels: persist model+experiment, inspect dataset asynchronously, starter-model confirmation, CPU/GPU availability, true phase/epoch metrics and data-aware stale measurements; retain hierarchy/Python/probes.
- [x] Integrate and review source CPU CSV+image runs, checkpoint replay, restore/cancel and existing S1 tests. Independently review split/resume/config/device semantics.
- [x] GPU validation only after available CUDA runtime/user setup decision; use a tiny fixture when resources permit, no benchmarks or long training. Otherwise retain explicit unverified/disabled state with exact reason.
- [x] Rebuild S2 EXE, run frozen file-based training and native framebuffer/callback checks, compare against native reference, update source/portable archives, guides, verification and handoff. Keep S1 package available.

## Acceptance gates

Data paths and feature/class selection survive save/reopen; graph/data/device errors stop before training; CSV and ImageFolder execute real train/val/test; same checkpoint independently reproduces inference; epoch-boundary CPU resume matches continuous training; worker failures/cancel terminate and retain atomic artifacts; no model/setting changes are hidden behind stale metadata. Frozen EXE must exercise the same data/training profile as source. Report actual GPU evidence and remaining boundaries without substituting menus for implementation.

## Final evidence

See studio/native/S2_VERIFICATION.md. Backend122 and native38 tests passed in each runtime; CPU and cuda:0 source/frozen diagnostics passed. GPU checks use generated CSV/PNG fixtures, not performance benchmarks. GitHub remains deferred.
