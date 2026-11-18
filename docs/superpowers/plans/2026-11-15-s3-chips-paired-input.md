# S3 Native Chips and Paired Inputs Implementation Plan

> **For agentic workers:** Use subagent-driven development, the shared studio/S3_CONTRACT.md and isolated file ownership. Test behavior first, observe its failure, implement, independently review, then integrate.

**Goal:** Extend the existing native workbench to editable/versioned multi-output chips and actual image+tabular single-device experiments, preserving S1/S2.

**Architecture:** Extend the existing graph edge contract and static Python profile; a paired Dataset adapter feeds named model inputs through the current classification worker. Document commands and local chip bundles stay UI-independent, while native controls adapt those commands.

**Approval:** User approved the previous four-part roadmap and said “推进”. This plan executes its first deliverable. Existing product design/reference remains; no further ideation/approval gate. No Git HEAD/identity exists, so do not fabricate commits or worktrees. No Use Computer or GitHub actions.

- [x] Named-output core/Python: qualified refs, explicit multi-output routing/shapes, preserved sharing/state/budgets, static dict return/indexing, compatible old profiles. Independent reference and negative/edited-source/metadata-free tests.
- [x] Paired data/runtime: validated config, audited source/group/split/sample pairing, dict batches, named port mapping, compatible legacy checkpoints. Small file fixtures, no clinical data; CPU exact resume and selected CUDA verification.
- [x] Document/chip bundles: atomic interface/public-parameter commands, named ports, bounded semantic undo/redo, nested encapsulation, immutable versioned library/closure/collision import. Tests before edits, preserve incomplete drafts and old APIs.
- [x] Native controls/workspace: paired settings and starter; multi-output sockets/probes; chip interface/public parameter/library controls; undo/redo. Async scans and native callback/framebuffer checks, same style.
- [x] Integrate and independently review all numerical, binding, sharing, leakage and restore boundaries. Fix substantive findings with regressions.
- [x] Preserve S2 archives, build CPU/CUDA S3 EXEs and verify actual paired chip training, source roundtrip/reopen/probes. Update guides/evidence/source/portable archives, keep upload paused.

The final acceptance is an actual paired fusion experiment using a nested multi-output custom chip, with CPU/GPU execution and independently reproduced predictions—not just new UI controls. Scope details and exact APIs are authoritative in studio/S3_CONTRACT.md.

2026-11-15 evidence: each runtime ran 175 backend tests (1 skipped for Windows symlink privilege), and 67 native tests passed in each build environment. Ten source/frozen runtime/render diagnostics passed with stopped workers. CPU/CUDA EXEs and portable ZIP CRC/SHA256 verified, including CPU selection inside the CUDA EXE. See studio/native/S3_VERIFICATION.md and docs/评审附件/2026-11-15-S3验收记录.json. The source archive passed CRC, every archived entry matched its current local file, and all 64 indexed Python source hashes matched; final archive identity is in output/native-s3/archive-source.json, outside the archive to avoid a circular self-hash. GitHub upload remains paused.
