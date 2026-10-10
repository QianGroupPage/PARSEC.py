"""Archive unchanged actual Poisson systems during a full-SCF control run."""
import sys
from pathlib import Path
import numpy as np
from unittest.mock import patch
from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend as Backend
from parsec_python.acceleration.cli import main as cli_main

def main():
    directory=Path(sys.argv.pop(1)).resolve(); directory.mkdir(parents=True,exist_ok=True)
    original=Backend.solve; counter=0
    def solve(self,rhs,initial,**kwargs):
        nonlocal counter
        result=original(self,rhs,initial,**kwargs)
        if counter in (0,5,25):
            a=self.operator
            np.savez(directory/f'poisson_{counter}.npz',data=a.data,indices=a.indices,indptr=a.indptr,
                     shape=a.shape,rhs=rhs,initial=initial,solution=result['solution'],
                     rtol=kwargs['relative_tolerance'],atol=kwargs['absolute_tolerance'],
                     max_iterations=kwargs['max_iterations'],iterations=result['iterations'])
            print('CAPTURED_POISSON',counter,flush=True)
        counter+=1
        return result
    with patch.object(Backend,'solve',solve): return cli_main()

if __name__=='__main__': main()
