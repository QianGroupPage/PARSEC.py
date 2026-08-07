# Port provenance

This directory records what has actually been ported and verified. It exists
to prevent a working approximation, a compatibility wrapper, or a benchmark
match from being described as a literal PARSEC algorithm port.

`source_map.json` is the machine-readable source of truth. Each component
records:

- the concept package that will own the native Python implementation;
- the Python modules that implement the component;
- the PARSEC source files that must be audited;
- whether the concept package contains an implementation;
- whether literal algorithmic parity has been verified; and
- the evidence supporting any verification claim.

The concept packages own the native implementations, with no parallel flat
compatibility layer. Most components remain `implemented_unverified`. The
audited real CHEBFF/CHEBDAV/SUBSPACE slice is `partially_verified`, because
symmetry-representation behavior and deterministic cross-language
compiler-random/reduction trajectories remain incomplete.

## Status policy

`concept_package_status` may be:

- `scaffold_only`: package boundary and documentation exist, but no port is
  claimed;
- `in_progress`: native Python code is being developed;
- `implemented_unverified`: the intended algorithm exists in Python but has
  not passed the required source and numerical audits;
- `verified_component`: source mapping and focused numerical tests support
  component parity;
- `verified_integration`: component and end-to-end reference tests support
  parity; or
- `out_of_scope`: intentionally excluded from the declared calculation scope.

`literal_parsec_status` remains `not_verified` until the corresponding PARSEC
routine, defaults, state transitions, stopping tests, and failure behavior
have all been checked. A close final total energy alone is not sufficient.

## Native-port boundary

The port must not execute or link to PARSEC through `subprocess`, WSL,
`ctypes`, `cffi`, `f2py`, a shared library, or any other interface. NumPy and
SciPy may provide general array, sparse-matrix, QR, and dense eigensystem
primitives, but selecting a PARSEC algorithm must execute that algorithm's
Python control flow. In particular, a Chebyshev solve must not silently fall
back to ARPACK.

When a component is promoted, update both the manifest entry and its evidence
list in the same change.
