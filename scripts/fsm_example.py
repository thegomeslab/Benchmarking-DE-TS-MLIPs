# ruff: noqa: PLC0415
"""
Example script for running the Freezing String Method (FSM).

Users must install their desired quantum chemistry backend separately from the
mlfsm package. Currently supported calculators include:

    - QChem
    - xTB (GFN2-xTB)
    - FAIR UMA
    - AIMNet2
    - MACEOFF
    - SchNet
    - EMT

Only the selected calculator needs to be installed in the Python environment.
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime


from mlfsm.cos import FreezingString
from mlfsm.geom import project_trans_rot
from mlfsm.opt import CartesianOptimizer, InternalsOptimizer, Optimizer
from mlfsm.utils import load_xyz

from counting_calculator import CountingCalculator


HERE = Path(__file__).parent


def run_fsm(
    reaction_dir: Path | str,
    optcoords: str = "cart",
    interp: str = "lst",
    method: str = "L-BFGS-B",
    maxls: int = 3,
    maxiter: int = 1,
    dmax: float = 0.3,
    nnodes_min: int = 10,
    ninterp: int = 100,
    suffix: str | None = None,
    calculator: str = "qchem",
    chg: int = 0,
    mult: int = 1,
    nt: int = 64,
    verbose: bool = False,
    ckpt: Path = HERE / "pre_trained_gnns/schnet_fine_tuned.ckpt",
    interpolate: bool = False,
    **kwargs,
):
    """Run the Freezing String Method on a given reaction with user specified parameters."""
    reaction_dir = Path(reaction_dir)

    if suffix:
        outdir = reaction_dir / (
            f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_"
            f"maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}_{suffix}"
        )
    else:
        outdir = (
            reaction_dir
            / f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}"
        )

    if interpolate:
        outdir = reaction_dir / f"interp_{interp}"

    if outdir.is_dir():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # Load structures
    reactant, product = load_xyz(reaction_dir)
    with open(os.path.join(reaction_dir,"chg")) as f:
              chg = int(f.read())
    with open(os.path.join(reaction_dir,"mult")) as f:
              mult = int(f.read())
    
    print(chg)
    print(mult)
    name = str(reaction_dir).split('/')[-1]
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
    calc: Any

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
            label=f"{name}",
            method="b3lyp",
            basis="def2-svp",
            charge=chg,
            multiplicity=mult,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=nt,
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
    elif calculator == "torchmd":
        from torchmd_calc import TMDCalculator

        #Check charge and multiplicity
        if mult != 1:
            raise ValueError(f"This TensorNet model only can use systems of multiplicity 1")
        if chg != 0:
            raise ValueError(f"This TensorNet model only can use systems of charge 0")
        calc = TMDCalculator()
        calc = CountingCalculator(calc)
    elif calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE  # type: ignore [import-not-found]

        calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
        calc = CountingCalculator(calc)
    elif calculator == "emt":
        from ase.calculators.emt import EMT

        calc = EMT()
        calc = CountingCalculator(calc)
    elif calculator == "mace":
        import torch
        from mace.calculators import mace_off  # type: ignore [import-not-found]

        #Check charge and multiplicity
        if mult != 1:
            raise ValueError(f"MACEOFF23 model only can use systems of multiplicity 1")
        if chg != 0:
            raise ValueError(f"MACEOFF23 model only can use systems of charge 0")
        calc = TMDCalculator()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(model="large", device=dev)
        calc = CountingCalculator(calc)
    elif calculator == "pm6":
        from pm6ml import PM6MLCalculator  # type: ignore [import-not-found]

        calc = PM6MLCalculator(model_file = "/Users/jonmarks/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt")
        calc = CountingCalculator(calc)
    elif calculator == "maceomol":
        import torch
        from mace.calculators import mace_omol

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_omol(model="extra_large",device=dev)
        calc = CountingCalculator(calc)
    else:
        raise ValueError(f"Unknown calculator {calculator}")

    # Initialize FSM string
    string = FreezingString(reactant, product, nnodes_min, interp, ninterp)
    if interpolate:
        string.interpolate(outdir)
        return

    #align reactant and product structures
    _,aligned_prod = project_trans_rot(reactant.get_positions(),product.get_positions())
    product.set_positions(aligned_prod.reshape(-1,3))
    
    optimizer: Optimizer
    # Choose optimizer
    if optcoords == "cart":
        optimizer = CartesianOptimizer(calc, method, maxiter, maxls, dmax)
    elif optcoords == "ric":
        optimizer = InternalsOptimizer(calc, method, maxiter, maxls, dmax)
    else:
        raise ValueError("Check optimizer coordinates")

    # Run FSM
    while string.growing:
        string.grow()
        string.optimize(optimizer)
        string.write(outdir)
    print(f"Calculator counted: {calc.grad_calls}")
    print(f"Gradient calls: {string.ngrad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction_dir", type=Path, help="absolute path to reaction")
    parser.add_argument(
        "--optcoords", type=str, default="cart", choices=["cart", "ric"], help="Coordinates for optimization"
    )
    parser.add_argument(
        "--interp", type=str, default="ric", choices=["cart", "lst", "ric"], help="Interpolation method"
    )
    parser.add_argument("--nnodes_min", type=int, default=18, help="Minimum number of nodes in the FSM string")
    parser.add_argument("--ninterp", type=int, default=50, help="Number of interpolation points between nodes")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for output directory")
    parser.add_argument(
        "--method", type=str, default="L-BFGS-B", choices=["L-BFGS-B", "CG"], help="Optimization method"
    )
    parser.add_argument("--maxls", type=int, default=3, help="Maximum number of line search iterations")
    parser.add_argument("--maxiter", type=int, default=2, help="Maximum number of optimization iterations")
    parser.add_argument("--dmax", type=float, default=0.05, help="Maximum displacement for optimization steps")
    parser.add_argument(
        "--calculator",
        type=str,
        default="qchem",
        choices=["qchem",'b3lyp', "xtb", "schnet","pm6", "torchmd", "uma_s","uma_m","eSEN","aimnet2","mace", "emt","maceomol"],
        help="Calculator to use for energy and gradient evaluations",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=HERE / "pre_trained_gnns/schnet_fine_tuned.ckpt",
        help="Checkpoint for calculator",
    )
    parser.add_argument("--chg", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--nt", type=int, default=64, help="Number of threads for the calculator")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--interpolate", action="store_true", help="Run interpolation instead of FSM")

    args = parser.parse_args()
    run_fsm(**vars(args))
