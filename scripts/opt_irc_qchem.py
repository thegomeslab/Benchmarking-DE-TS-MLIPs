import os
import sys
from ase.io import read, write
from sella import IRC
from ase.calculators.qchem import QChem
from xtb.ase.calculator import XTB
import torch
from fairchem.core import FAIRChemCalculator, pretrained_mlip 


if __name__ == '__main__':

    xyzfile = sys.argv[1]        
    atoms = read(xyzfile)
    with open("./chg",'r') as f:
        chg = int(f.read())
    with open("./mult",'r') as f:
        mult = int(f.read())
    # atoms.calc = QChem(label="irc", method='wb97x-v', basis='def2-tzvp', charge=chg, multiplicity=mult,
    #                                  scf_algorithm='diis',
    #                                  scf_max_cycles='500', nt=32)
    #atoms.calc = XTB(method="GFN2-xTB")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.calc = calc
    dirname, fname = os.path.split(xyzfile)
    outfile = os.path.join(dirname, "irc_f_"+fname+".traj")
    opt = IRC(atoms, trajectory=outfile, dx=0.3, eta=1e-4, gamma=0.1, keep_going=True)
    opt.run(fmax=0.1, steps=50, direction='forward')
    f_atoms = read(outfile, index=':')
    opt.run(fmax=0.1, steps=50, direction='reverse')
    b_atoms = read(outfile, index=':')[len(f_atoms):]

    irc_atoms = f_atoms[::-1] + b_atoms[1:]

    outfile = os.path.join(dirname, "irc_"+fname)
    write(outfile, irc_atoms, format='xyz')

