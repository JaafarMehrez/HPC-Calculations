# Author: Jaafar Mehrez, jaafar@hpqc.org
import numpy as np
import pyscf
from pyscf import cc, lib, tools, scf, symm, ao2mo
from pyscf.tools import fcidump
from pyscf.tools.fcidump import from_mo
from pyscf.tools.fcidump import from_integrals
from pyscf.gto.basis import parse_gaussian
import pyscf.symm.param as param
import pyscf.lib.logger as logger

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

name = 'out'
mol = pyscf.M(
    atom = '''
        LI
    ''',
    unit = 'angstrom',
    basis = {
            'LI' : parse_gaussian.load('STO-2G.gbs', 'LI')
    },
    charge = 0,
    spin = 1,
    verbose = 9,
    symmetry = True,
    output = name +'.txt',
    symmetry_subgroup = 'D2h',
    max_memory = 4000,
)

mf = mol.ROHF().set(conv_tol=1e-10,max_cycle=999,ddm_tol=1e-14,direct_scf_tol=1e-14,chkfile=name+'.chk',init_guess='atom',irrep_nelec={'Ag': 3})
mf.kernel()
pyscf.tools.fcidump.from_chkfile('fort.55',name+'.chk',tol=1e-18,float_format='% 0.20E',molpro_orbsym=False,orbsym=None)
