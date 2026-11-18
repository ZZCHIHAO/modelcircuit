# S6 CSV regression implementation plan

**Release gate.** A CSV with one continuous target can be inspected, wired, trained, evaluated and resumed from the native workbench on CPU and one selected CUDA device. The release is v0.6.0-alpha.1.

1. Define `regression-v1` and `regression_csv` in the experiment contract. Add a deterministic row/group split and reuse CSV parsing, source verification and training-only scaling. Test deliberate leakage, non-finite targets, constant-target metrics and frozen inputs.
2. Add a regression task adapter for float32 `[batch, 1]` named `prediction`, MSE loss, MAE, RMSE and R². Share the supervised worker and checkpoint machinery. Retain classification artifact/identity formats. Test data mutation, wrong output contracts and epoch continuation against uninterrupted training.
3. Connect the native task selector, data modal, starter graph and result view. Verify save/reopen, model Python round-trip, worker result, probes and cancellation using application callbacks and a local fixture.
4. Run the backend and native suites, verify CPU and CUDA with independent checkpoint inference, build isolated S6 desktop distributions and smoke-test them. Update the usage notes and changelog, then verify the private pull request and uploaded tree.

## Additional catalog deliverable

- Partition the built-in library by basic operators, deep-learning models, classical estimators and architecture/fusion chips. Retain search, drag, port display and settings on the existing palette.
- Deliver a single-estimator SVC/SVR fit path on CPU alongside the Torch epoch loop. Verify predictions against independently fitted scikit-learn models, record artifacts, and test the generated Python import contract.
