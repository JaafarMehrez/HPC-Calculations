# Author: Jaafar Mehrez, jaafar@hpqc.org
import numpy as np
import pyscf
from pyscf import cc, lib, tools, scf, symm, ao2mo
from pyscf.tools import fcidump
from pyscf.gto.basis import parse_gaussian

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, np.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(f" {nmo:4d} {nelec:2d}\n")
    if orbsym is not None and len(orbsym) > 0:
        orbsym = [x + 1 for x in orbsym]
        fout.write(f"{' '.join([str(x) for x in orbsym])}\n")
    else:
        fout.write(f"{' 1' * nmo}\n")
    fout.write(' 150000\n')

fcidump.write_head = write_head
R_CO = 0.5
Molecule_ZMAT = f'''
C1
O1 1 {R_CO}
'''
name = 'out'
mol = pyscf.M( 
    atom = Molecule_ZMAT,
    unit = 'angstrom',
    basis = {'C' : parse_gaussian.load('C-aVDZ-EMSL.gbs', 'C'), 
             'O' : parse_gaussian.load('O-aVDZ-EMSL.gbs', 'O')},
    charge = 0,
    spin = 0,
    symmetry = True,
    verbose = 9,
    output = name +'.txt',
    max_memory = 4000,
)
mf = mol.RHF().set(conv_tol=1e-10,max_cycle=999,direct_scf_tol=1e-15, chkfile=name+'.chk', init_guess='atom')
mf.kernel()
pyscf.tools.fcidump.from_chkfile('fort.55',name+'.chk',tol=1e-18, float_format='% 0.20E',molpro_orbsym=False,orbsym=None)
