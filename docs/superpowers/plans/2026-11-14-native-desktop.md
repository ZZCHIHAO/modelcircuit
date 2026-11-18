# Native Desktop S1 Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans. The user explicitly selected the original ML Forge desktop form; preserve that direction while implementing the existing S1 semantics.

**Goal:** Deliver a Windows executable using DearPyGui native windows/node editors, the existing independent S1 core and real CPU workers. After acceptance, upload the project to the user's private GitHub repository with the strongest available PR-only collaboration policy that meets their account capabilities.

**Architecture:** Desktop UI calls a pure Python Document editor and existing RunManager directly. No browser, Vite server or HTTP coordinator is required for desktop operation. The frozen main entry calls multiprocessing.freeze_support before importing/initializing the GUI. Runtime data lives in LocalAppData; bundled assets are read-only.

**Tech stack:** DearPyGui 2.3.1, existing PyTorch 2.6.0+cpu, PyInstaller 6.22.3. Reuse the original icon and native layout conventions; keep old ml_forge business sources and the web workbench intact.

## Tasks

- [x] Pure Document editor: native/document.py plus tests for scope-aware edits, sharing, boundary inference, source round-trip and precise probes. Core implementer owns these files.
- [x] DearPyGui workbench: native/ui.py using original left palette, middle canvas, right properties/training and bottom measurements. UI implementer owns GUI files. All callbacks/DPG updates run on main thread; results come from workers only.
- [x] Entry/lifecycle: native/main.py, native/paths.py, native/acceptance.py. Root owns resource lookup, persistent paths/logs, freeze_support, worker cleanup, CLI diagnostic/self-test modes and structured reports.
- [x] Packaging: pinned requirements-desktop.txt, native/MLForgeStudio.spec, build-desktop.ps1 and desktop launcher. Include actual torch runtime/DLLs and original icon; prefer a portable directory containing an EXE over slow self-extracting onefile mode. Do not imply a directory build is a solitary self-contained file.
- [x] Native validation: new document tests and existing core tests; real source and frozen worker execution; real DPG window rendering, screenshots, close/cancel cleanup, Chinese-path and arbitrary-cwd operation. Record tool limitations honestly.
- [x] Update README, native acceptance evidence and the selected-form correction. Retain the full-platform/S1 boundary and real synthetic-data labels.
- [ ] Inspect authenticated GitHub CLI/app access without exposing credentials; create/push a private repository only after local acceptance. Exclude dependencies, built binaries unless explicitly publishing a release, local logs/runs, machine state and secrets; include source, lock/pins, license attribution and reproducible build instructions. Apply/test available branch and collaboration controls; never claim personal-repo collaborators are read-only if GitHub grants write access.

## Acceptance

Double-click MLForgeStudio.exe from its portable folder to open one native window without Python/Node command-line setup. Connect named ports, inspect/share/expand chips, edit Python within the S1 subset, save JSON, run the actual seeded CPU experiment, inspect measured probes and cancel/exit with child-process cleanup. The exe builds and runs locally; clean-machine compatibility, code signing and installer distribution are separate claims requiring evidence.

The original screenshot is the native interaction/style reference. This change supersedes the tentative WebView wrapper investigation; no WebView implementation or dependency is introduced.

## Final local acceptance and deferred upload

62 backend tests and 17 native tests passed. The actual frozen EXE passed CPU/Python/shared-probe/cancellation diagnostics and native DPG framebuffer/callback checks with all workers stopped. Sources, screenshots and exact run IDs are recorded in `studio/native/VERIFICATION.md` and `design-qa.md`.

The user explicitly stopped Use Computer debugging; remaining validation used only code, application-owned rendering/callbacks and frozen diagnostics. GitHub CLI was not authenticated, and the user's Tabbit session could not reach GitHub (connection reset/timeouts). No remote repository or upload occurred. The user explicitly deferred upload to a later new conversation. `studio/HANDOFF.md` records the exact personal-private/owner-only/no-collaborator policy, tooling, source staging and recovery instructions. No future automation was created.
