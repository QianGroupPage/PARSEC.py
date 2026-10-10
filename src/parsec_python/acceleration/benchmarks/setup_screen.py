"""Measure preparation through reduced-operator construction, then stop.

Does not claim a full-program timing. A cold test needs no flag: the CLI uses
no symmetry cache unless --symmetry-cache names a directory.
"""
import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
import parsec_python.acceleration.Eigensolvers.symmetry as symmetry
from parsec_python.acceleration.cli import main as cli_main

def main():
    dest=Path(sys.argv.pop(1)).resolve()
    allocated_affinity=sorted(os.sched_getaffinity(0))
    started=time.perf_counter(); original=symmetry.load_or_build_reduced_operators
    def stop(*a,**kw):
        t=time.perf_counter();result=original(*a,**kw)
        row=dict(total_preparation_to_operators_s=time.perf_counter()-started,
                 operator_s=time.perf_counter()-t,threads=os.environ.get('OMP_NUM_THREADS'),
                 cache_disabled=not result.cache_info.enabled,
                 sectors=len(result.stencil_metadata))
        # OpenMP binding may narrow the calling thread's mask to one core.
        # Keep that separate from the process mask before native work starts.
        row['initial_cpu_affinity']=allocated_affinity
        row['calling_thread_affinity_after_openmp']=sorted(os.sched_getaffinity(0))
        from parsec_python.acceleration.backends.native import native_build_info
        row['native_thread_configuration']=native_build_info()
        topology=Path('/sys/devices/system/cpu')
        row['physical_cores_in_initial_affinity']=len({((topology/f'cpu{i}/topology/physical_package_id').read_text().strip(),
            (topology/f'cpu{i}/topology/core_id').read_text().strip()) for i in allocated_affinity})
        dest.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);raise SystemExit(0)
    with patch.object(symmetry,'load_or_build_reduced_operators',stop):return cli_main()

if __name__=='__main__':main()
