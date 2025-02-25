import pyscf
from pyscf import cc, lib, tools, scf, symm, ao2mo
from pyscf.gto.basis import parse_gaussian
from pyscf.cc.mrcc import MRCCInterface

name = 'out'
mol = pyscf.M(
    atom = '''
        O
    ''',
    unit = 'angstrom',
    basis = {
            'O' : parse_gaussian.load('O-aCVQZ-EMSL.gbs', 'O')
    },
    charge = 1,
    spin = 3,
    symmetry = True,
    verbose = 9,
    symmetry_subgroup = 'D2h',
    output = name +'.txt',
    max_memory = 8000,
)
mf = mol.UHF().set(conv_tol=1e-11,max_cycle=9999,ddm_tol=1e-13,direct_scf_tol=1e-13,chkfile=name+'.chk',init_guess='atom',irrep_nelec={'Ag': 4, 'B3u':1 , 'B2u':1 ,'B1u':1 })
mf.kernel()

#pyscf.tools.fcidump.from_chkfile_uhf('FCIdump',name+'.chk',tol=1e-18, float_format='% 0.20E',molpro_orbsym=False,orbsym=None)
pyscf.tools.fcidump.from_scf_uhf(mf,'fort.55',tol=1e-18, float_format='% 0.20E',molpro_orbsym=False)

#MRCCInterface.run_mrcc(mf, fort_file='fort.55', mrcc_input_file='MINP', tol=1e-18, float_format='% 0.20E', molpro_orbsym=False)
