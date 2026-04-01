import argparse
import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import ase.db
from ase.calculators.qchem import QChem
from ase.io import read, write
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize.bfgs import BFGS
from mlfsm.interp import RIC
from counting_calculator import CountingCalculator


parser = argparse.ArgumentParser()

parser.add_argument("output",type=str,help="absolute path to reaction")
parser.add_argument("--transition_state", type=str, default=None)
parser.add_argument("--n_images", type=int, default=7)
parser.add_argument("--neb_fmax", type=float, default=0.1)
parser.add_argument("--cineb_fmax", type=float, default=0.05)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
        "--calculator",
        type=str,
        default="qchem",
        choices=["qchem", "xtb", "torchmd", "uma_s","uma_m","eSEN","aimnet2","mace","maceomol","emt"],
        help="Calculator to use for energy and gradient evaluations",
    )
args = parser.parse_args()

def project_trans_rot(a, b):
    centroid_a = np.mean(a, axis=0, keepdims=True)
    centroid_b = np.mean(b, axis=0, keepdims=True)
    A = a - centroid_a
    B = b - centroid_b
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = centroid_b@R - centroid_a
    return a.flatten(), (b@R-t).flatten()

def load_xyz(reaction_dir):

    xyz = os.path.join(reaction_dir, "initial.xyz")
    if not os.path.exists(xyz):
        raise Exception(f"Input file {xyz} not found.")
    atoms = read(xyz, format="xyz", index=":")
    reactant = atoms[0]
    product = atoms[-1]
    r_xyz, p_xyz = project_trans_rot(reactant.get_positions(), product.get_positions())
    reactant.set_positions(r_xyz.reshape(-1, 3))
    product.set_positions(p_xyz.reshape(-1, 3))
    return reactant, product

def main(args):  # pylint: disable=redefined-outer-name

    reaction_dir = args.output
    # load charge and mult
    chgfile = os.path.join(os.getcwd(), reaction_dir, "chg")
    multfile = os.path.join(os.getcwd(), reaction_dir, "mult")
    with open(chgfile, 'r') as f:
        chg = int(f.read())
    with open(multfile, 'r') as f:
        mult = int(f.read())
    # Load structures
    reactant, product = load_xyz(reaction_dir)
    reactant.info.update({'charge': chg, 'spin': mult}) #works for FAIRchem models
    product.info.update({'charge': chg, 'spin': mult})
    
    #update the actual ase information
    chg_list = reactant.get_initial_charges()
    chg_list[0] = chg
    mult_list = reactant.get_initial_magnetic_moments()
    mult_list[0] = mult-1
    
    reactant.set_initial_magnetic_moments(mult_list)
    reactant.set_initial_charges(chg_list)
    
    product.set_initial_magnetic_moments(mult_list)
    product.set_initial_charges(chg_list)
    
    print(f"CHG:\n{chg_list} mult: {mult_list}")
    par = True
    interp = RIC(reactant,product,ninterp=args.n_images)
    interpolated_positions=interp()

    atom_configs = [reactant.copy() for i in range(args.n_images - 1)] + [product]

    # set calculator
    if args.calculator == "qchem":
        from ase.calculators.qchem import QChem
        for i, (pos,atom_config) in enumerate(zip(interpolated_positions,atom_configs)):
            label = os.path.join(reaction_dir,"calc",f"image_{i}")
            os.makedirs(os.path.dirname(label),exist_ok=True)
            first_calc = QChem(
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
            setattr(first_calc, "_prefix", label)
            print(f"[DEBUG] Q-Chem calculator prefix for image {i} BEFORE wrapping:", first_calc.prefix)
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.calc = calculator
            print(f"Prefix for image {i}: ", atom_config.calc.prefix)
    elif args.calculator == "xtb":
        from xtb.ase.calculator import XTB
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            first_calc = XTB(method='GFN2-xTB')
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.calc = calculator
    elif args.calculator == "uma_s":
        import torch
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=dev)
            first_calc = FAIRChemCalculator(predictor, task_name="omol")
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.info.update({"spin": mult, "charge": chg})
            atom_config.calc = calculator
        par = False
    elif args.calculator == "uma_m":
        import torch
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
            first_calc = FAIRChemCalculator(predictor, task_name="omol")
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.info.update({"spin": mult, "charge": chg})
            atom_config.calc = calculator
        par = False
    elif args.calculator == "eSEN":
        import torch
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device=dev)
            first_calc = FAIRChemCalculator(predictor)
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.info.update({"spin": mult, "charge": chg})
            atom_config.calc = calculator
        par = False 
    elif args.calculator == "torchmd":
        from torchmd_calc import TMDCalculator
        first_calc = TMDCalculator()
        calculator = CountingCalculator(first_calc)
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            atom_config.set_positions(pos)
            atom_config.calc=calculator
    elif args.calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE

        for pos,atom_config in zip(interpolated_positions,atom_configs):
            first_calc = AIMNet2ASE("aimnet2",charge=chg,mult=mult)
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.calc = calculator
    elif args.calculator == "mace":
        import torch
        from mace.calculators import mace_off
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            first_calc = mace_off(model='medium',device=dev)
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.calc = calculator
    elif args.calculator == "maceomol":
        import torch
        from mace.calculators import mace_omol
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for pos,atom_config in zip(interpolated_positions,atom_configs):
            first_calc = mace_omol(model='extra_large',device=dev)
            calculator = CountingCalculator(first_calc)
            atom_config.set_positions(pos)
            atom_config.calc = calculator
    else:
        raise Exception(f"Unknown calculator {calculator}")
    

    print("Running NEB ... ")
    neb = NEB(atom_configs, climb=True, parallel=par)
    neb.climb=False
    calculation_checker = CalculationChecker(neb)
    neb_tools = NEBTools(neb.images)

    relax_neb = NEBOptimizer(neb)
    fmaxs = []

    relax_neb.attach(calculation_checker.check_calculations)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))

    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    print("NEB has converged, turn on CI-NEB ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)
    
    outdir = os.path.join(args.output, f"neb_nimages_{args.n_images}_calculator_{args.calculator}_nebfmax_{args.neb_fmax}_cinebfmax_{args.cineb_fmax}_steps_{args.steps}_counting")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    if converged:
        open(os.path.join(outdir, "converged"), "w")
        print("Reaction converged ... ")
    json.dump(fmaxs, open(os.path.join(outdir, "fmaxs.json"), "w"))
    
    grad_calls = 0
    for structure in atom_configs:
        grad_calls += structure.calc.grad_calls
    with open(os.path.join(outdir, "grad_calls_neb.txt"),'w') as f: f.write(str(grad_calls))
        
    transition_state = max(atom_configs, key=lambda x: x.get_potential_energy())
    write(os.path.join(outdir, "neb_path.xyz"),atom_configs)
    write(os.path.join(outdir, "transition_state.xyz"), transition_state)
    write(os.path.join(outdir, "transition_state.png"), transition_state)
    write(os.path.join(outdir, "reactant.xyz"), atom_configs[0])
    write(os.path.join(outdir, "reactant.png"), atom_configs[0])
    write(os.path.join(outdir, "product.xyz"), atom_configs[-1])
    write(os.path.join(outdir, "product.png"), atom_configs[-1])


class CalculationChecker:
    def __init__(self, neb):
        self.neb = neb

    def check_calculations(self):
        missing_calculations = []
        for i, image in enumerate(self.neb.images[1:-1]):
            if {"forces", "energy"} - image.calc.results.keys():
                missing_calculations.append(i)

        #if missing_calculations:
        #    raise ValueError(f"missing calculation for image(s) {missing_calculations}")


if __name__ == "__main__":
    main(args)
