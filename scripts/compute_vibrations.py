import os
import sys
import numpy as np
from typing import Any
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from ase.vibrations import Vibrations
from counting_calculator import CountingCalculator

# Conversion constants
BOHR_TO_ANG = 0.529177210903  # bohr to angstrom
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG  # angstrom to bohr
EV_TO_HARTREE = 0.0367493  # eV to Hartree
# Hessian conversion: eV/Ang^2 to Hartree/bohr^2
HESSIAN_CONVERSION = EV_TO_HARTREE / (ANG_TO_BOHR ** 2)


def compute_hessian_with_ase(atoms, delta=0.01, vib_name='vib_temp'):
    """
    Compute numerical Hessian using ASE's finite difference approach.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached
    delta : float
        Displacement for finite differences in Angstrom
    vib_name : str
        Name/path for the vibrations directory
        
    Returns
    -------
    hessian : np.ndarray
        Hessian matrix in eV/Ang^2, shape (3*n_atoms, 3*n_atoms)
    """
    vib = Vibrations(atoms, delta=delta, name=vib_name)
    vib.run()
    
    # Get the VibrationsData object, then extract Hessian
    vib_data = vib.get_vibrations()
    hessian = vib_data.get_hessian_2d()
    
    # Don't clean up - keep the vibration files
    # vib.clean()
    
    return hessian


if __name__ == '__main__':
    xyzfile = sys.argv[1]
    calculator_name = sys.argv[2]
    reaction_dir = os.path.dirname(os.path.dirname(sys.argv[1]))
    atoms = read(xyzfile)
    
    # Read charge and multiplicity
    with open(os.path.join(reaction_dir, "chg")) as f:
        chg = int(f.read())
    with open(os.path.join(reaction_dir, "mult")) as f:
        mult = int(f.read())
    
    print(f"Charge: {chg}, Multiplicity: {mult}")
    atoms.info.update({'charge': chg, 'spin': mult})
    
    # Update ASE charge and magnetic moment info
    chg_list = atoms.get_initial_charges()
    chg_list[0] = chg
    mult_list = atoms.get_initial_magnetic_moments()
    mult_list[0] = mult - 1
    
    atoms.set_initial_magnetic_moments(mult_list)
    atoms.set_initial_charges(chg_list)
    
    calc: Any
    # Load calculator (same as your original code)
    if calculator_name == "qchem":
        from ase.calculators.qchem import QChem
        calc = QChem(
            label="fsm",
            method="wb97x-v",
            basis="def2-tzvp",
            charge=chg,
            multiplicity=mult,
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=48,
        )
    elif calculator_name == "xtb":
        from xtb.ase.calculator import XTB
        calc = XTB(method="GFN2-xTB")
        calc = CountingCalculator(calc)
    elif calculator_name == "uma_s":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        calc = CountingCalculator(calc)
    elif calculator_name == "uma_m":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
        calc = CountingCalculator(calc)
    elif calculator_name == "eSEN":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device=dev)
        calc = FAIRChemCalculator(predictor)
        calc = CountingCalculator(calc)
    elif calculator_name == "aimnet2":
        from aimnet2calc import AIMNet2ASE
        calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
        calc = CountingCalculator(calc)
    elif calculator_name == "emt":
        from ase.calculators.emt import EMT
        calc = EMT()
        calc = CountingCalculator(calc)
    elif calculator_name == "maceomol":
        import torch
        from mace.calculators import mace_omol
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_omol(model="extra_large", device=dev)
        calc = CountingCalculator(calc)
    else:
        raise ValueError(f"Unknown calculator {calculator_name}")
    
    atoms.calc = calc
    
    # Setup paths for output
    dirname, fname = os.path.split(xyzfile)
    vib_path = os.path.join(dirname, f"vib_{calculator_name}")
    
    # Compute Hessian using ASE
    print("Computing Hessian with ASE...")
    hessian_ase = compute_hessian_with_ase(atoms, delta=0.01, vib_name=vib_path)
    
    # Convert to geometric format
    # Coordinates: Ang to bohr
    coords_ang = atoms.get_positions().flatten()
    coords_bohr = coords_ang * ANG_TO_BOHR
    
    # Hessian: eV/Ang^2 to Hartree/bohr^2
    hessian_au = hessian_ase * HESSIAN_CONVERSION
    
    # Get element symbols and masses
    elem = atoms.get_chemical_symbols()
    masses = atoms.get_masses()  # in amu
    
    # Get energy
    energy_ev = atoms.get_potential_energy()
    energy_hartree = energy_ev * EV_TO_HARTREE
    
    # Import geometric's frequency_analysis
    try:
        from geometric.normal_modes import frequency_analysis
    except ImportError:
        print("Error: geometric package not found. Install with: pip install geometric")
        sys.exit(1)
    
    # Call geometric's frequency analysis
    print("Running geometric frequency analysis...")
    outfile = os.path.join(dirname, f"vibrations_{calculator_name}_geometric.txt")
    
    # frequency_analysis returns (freqs, normal_modes) according to docstring
    # but may return additional values, so unpack accordingly
    result = frequency_analysis(
        coords=coords_bohr,
        Hessian=hessian_au,
        elem=elem,
        mass=masses,
        energy=energy_hartree,
        temperature=300.0,
        pressure=1.0,
        verbose=1,
        outfnm=outfile,
        note=f"Calculated with {calculator_name}",
        normalized=True
    )
    
    # Handle different return formats
    if isinstance(result, tuple):
        if len(result) == 2:
            freqs, normal_modes = result
        else:
            # Take first two elements
            freqs = result[0]
            normal_modes = result[1]
    else:
        freqs = result
        normal_modes = None
    
    # Print results
    print("\nVibrational Frequencies (cm^-1):")
    print("=" * 50)
    for i, freq in enumerate(freqs):
        if freq < 0:
            print(f"Mode {i+1:3d}: {freq:10.2f}i cm^-1")
        else:
            print(f"Mode {i+1:3d}: {freq:10.2f} cm^-1")
    
    # Also save a simple summary
    summary_file = os.path.join(dirname, f"frequencies_{calculator_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"# Vibrational frequencies calculated with {calculator_name}\n")
        f.write(f"# Using geometric for proper projection of translations/rotations\n")
        f.write(f"# Molecule: {fname}\n")
        f.write(f"# Number of atoms: {len(atoms)}\n")
        f.write(f"# Energy: {energy_ev:.6f} eV ({energy_hartree:.6f} Hartree)\n")
        f.write("#\n")
        f.write("# Mode    Frequency (cm^-1)\n")
        f.write("# ----    -----------------\n")
        for i, freq in enumerate(freqs):
            if freq < 0:
                f.write(f"{i+1:5d}    {freq:10.2f}i\n")
            else:
                f.write(f"{i+1:5d}    {freq:10.2f}\n")
    
    print(f"\nResults written to:")
    print(f"  - {outfile} (detailed geometric output)")
    print(f"  - {summary_file} (frequency summary)")