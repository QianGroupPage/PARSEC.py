"""Capture an actual symmetry-sector operator, then stop before SCF.

This benchmark instrumentation does not change the production source.
"""
import json
import sys
from pathlib import Path
import numpy as np
import parsec_python.acceleration.Eigensolvers.symmetry as symmetry
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition as Decomposition
from parsec_python.acceleration.cli import main as cli_main
from unittest.mock import patch

def main():
    destination = Path(sys.argv.pop(1)).resolve()
    build = Decomposition.build.__func__
    grid_holder = {}
    def capture_grid(cls, grid, *args, **kwargs):
        grid_holder['grid'] = grid
        return build(cls, grid, *args, **kwargs)
    original = symmetry.load_or_build_reduced_operators
    def capture(decomposition, *args, **kwargs):
        bundle = original(decomposition, *args, **kwargs)
        metadata = bundle.stencil_metadata[0]
        rows = decomposition.reduction.representative_rows[decomposition.sector_orbit_indices(0)]
        xyz = grid_holder['grid'].integer_coordinates[rows]
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, coordinates=xyz, neighbors=metadata.neighbors,
                 codes=metadata.coefficient_codes, palette=metadata.coefficient_palette)
        print('CAPTURED_STENCIL', json.dumps(dict(rows=len(rows), slots=metadata.neighbors.shape[0],
                                               file=str(destination))), flush=True)
        raise SystemExit(0)
    with patch.object(Decomposition,'build',classmethod(capture_grid)), \
         patch.object(symmetry,'load_or_build_reduced_operators',capture):
        return cli_main()

if __name__ == '__main__':
    raise SystemExit(main())
