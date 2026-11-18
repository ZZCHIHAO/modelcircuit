# S1 Python bridge

This is the implementer's phase record. The final integrated acceptance on 2026-11-14 is in [VERIFICATION.md](VERIFICATION.md): 25 bridge tests within 61 backend tests passed. Browser source editing and import rejection are recorded separately in `frontend/design-qa.md`.

Implementation: `backend/studio_core/python_io.py`.
Regression tests: `backend/tests/test_python_io.py`.

`export_python(graph)` validates the graph, then emits a standalone `Model`
and one ordinary `nn.Module` class per chip definition. Instances live in
`self.components = nn.ModuleDict(...)`; each shared instance is constructed
once and called through explicit named tensor values. Native PyTorch modules
receive their actual `input` keyword; chip and fusion ports retain their names.
State dictionary paths therefore use the same `components.<instance>` hierarchy
as the runtime. Frozen instances emit `requires_grad_(False)`.

`import_python(code)` uses only Python AST parsing and literal evaluation.
It does not execute source, import uploaded modules, invoke model constructors,
or use a graph cached in metadata. Current constructor literals/public parameter
bindings, component calls, value aliases, return values, Dropout probabilities,
Concat dimensions, and LayerNorm dimensions are reconstructed from current code.
Nested chip definitions share a registry while component IDs remain local to
their class. Standard torch/nn import aliases are recognized.

Generated forward arguments carry inert string shape annotations, for example
`a: 'float32[?,4]'`. These preserve shape/rank without visual metadata. Constructor
constraints determine a changed last input dimension. Optional visual metadata
restores display names and call positions only when their IDs/connections still
agree; it cannot change semantic shapes, dtypes, instances, parameters, or edges.

The supported profile is intentionally restricted: explicit ModuleDict model
classes; literal parameters or direct public constructor parameter bindings;
one tensor return per chip and named dictionary outputs at the root. Dynamic
control flow, arbitrary expressions, custom operators, modified Add/Concat helper
bodies, arbitrary top-level statements/imports, unknown syntax, and opaque
arbitrary-Python preservation are unsupported and raise readable `ValueError`s.
Input IDs must be valid non-keyword Python argument names and cannot begin with
`call_`. Bindings target immediate component parameters (including a nested chip's
public parameter). Parameter edits that make tensor dimensions inconsistent are
preserved as edits; callers can use the shared validator to inspect diagnostics.

TDD evidence: the initial suite produced ten expected failures because the bridge
was absent. Added shape/metadata, import-alias, native input-keyword, and LayerNorm
tuple cases were observed failing before their implementation. After the shared
CPU PyTorch installation, the latest run passed all 19 cases with no skips.
Both standalone tests launch isolated `python -I` subprocesses. The numerical
parity case uses two chip levels, public parameter overrides changing the inner
Linear weight shape from `[8,4]` to `[6,5]`, two calls to one shared outer encoder,
and a frozen classifier head. It loads the runtime's actual state dictionary
strictly into the standalone model and compares both named outputs, every
parameter gradient, both input gradients, and the frozen/trainable flags at
`rtol=1e-6, atol=1e-7`. All comparisons passed.

The numerical test's sensitivity was checked with a temporary generated-source
mutation changing Add into subtraction. That run failed with all six logits
mismatching (greatest absolute difference about 0.623). The mutation was removed
before the passing full run; it was never applied to production source.

Run the complete bridge suite:

```powershell
$env:PYTHONPATH = 'studio/backend'
& studio/.venv/Scripts/python.exe -m unittest discover -s studio/backend/tests -p test_python_io.py -v
```

The standalone test writes only generated, trusted code to a temporary directory
and launches Python with `-I`, away from the application package. Uploaded source
is never executed by import tests.
