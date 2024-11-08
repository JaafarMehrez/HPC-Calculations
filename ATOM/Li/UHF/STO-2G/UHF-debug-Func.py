import pyscf
from pyscf.gto.basis import parse_gaussian
from pyscf import ao2mo

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
mol.max_memory =4000
mf = mol.UHF().set(conv_tol=1e-10,max_cycle=999,direct_scf_tol=1e-15,chkfile=name+'.chk',init_guess='atom',irrep_nelec={'Ag': 3})
mf.kernel()
pyscf.tools.fcidump.write_uhf_integrals_fort55(mf,mol)
