import os
import sys
import torch
from typing import Any
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from sella import Sella
from torch import Tensor
from datetime import datetime
import time
from counting_calculator import CountingCalculator

if __name__ == '__main__':

    xyzfile = sys.argv[1]
    calculator = sys.argv[2]
    reaction_dir = os.path.dirname(os.path.dirname(sys.argv[1]))
    atoms = read(xyzfile)
    with open(os.path.join(reaction_dir,"chg")) as f:
        chg = int(f.read())
    with open(os.path.join(reaction_dir,"mult")) as f:
        mult = int(f.read())
    
    print(chg)
    print(mult)
    atoms.info.update({'charge': chg, 'spin': mult}) #works for FAIRchem models
    
    #update the actual ase information
    chg_list = atoms.get_initial_charges()
    chg_list[0] = chg
    mult_list = atoms.get_initial_magnetic_moments()
    mult_list[0] = mult-1
    
    atoms.set_initial_magnetic_moments(mult_list)
    atoms.set_initial_charges(chg_list)
    
    print(f"CHG:\n{chg_list} mult: {mult_list}")
    calc: Any
    label = reaction_dir.split('/')[-1].split('/')[-1]
    # Load calculator
    if calculator == "qchem":
        from ase.calculators.qchem import QChem

        calc = QChem(
            label="fsm",
            method="wb97x-v",
            basis="def2-tzvp",
            charge=chg,
            multiplicity=mult,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=nt,
        )
    elif calculator == "b3lyp":
        from ase.calculators.qchem import QChem

        calc = QChem(
            label=label,
            method="b3lyp",
            basis="def2-svp",
            charge=chg,
            multiplicity=mult,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=32,
        )

    elif calculator == "xtb":
        from xtb.ase.calculator import XTB  # type: ignore [import-not-found]

        calc = XTB(method="GFN2-xTB")
        calc = CountingCalculator(calc)
    elif calculator == "uma_s":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        calc = CountingCalculator(calc)
    elif calculator == "uma_m":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        calc = CountingCalculator(calc)
    elif calculator == "eSEN":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device=dev)
        calc = FAIRChemCalculator(predictor)
        calc = CountingCalculator(calc)
    elif calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE  # type: ignore [import-not-found]

        calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
        calc = CountingCalculator(calc)
    elif calculator == "emt":
        from ase.calculators.emt import EMT

        calc = EMT()
        calc = CountingCalculator(calc)
    elif calculator == "maceomol":
        import torch
        from mace.calculators import mace_omol

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_omol(model="extra_large",device=dev)
        calc = CountingCalculator(calc)
    else:
        raise ValueError(f"Unknown calculator {calculator}")
    atoms.calc = calc
    dirname, fname = os.path.split(xyzfile)
    outfile = os.path.join(dirname, "opt_"+fname+".traj")

    dyn = Sella(
        atoms,
        trajectory=outfile,
    ) 
    dyn.run(5e-3, 50)
    
    opt_atoms = read(outfile, index=':')
    outfile = os.path.join(dirname, "opt_"+fname)
    write(outfile, opt_atoms)
    write(os.path.join(dirname,"sella_ts_guess.xyz"),opt_atoms[-1],format="xyz")