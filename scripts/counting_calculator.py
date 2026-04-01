"""
Module providing CountingCalculator, a wrapper for ASE Calculators that
counts how many times gradients have been computed.

Author: Jonah Marks
"""

from typing import Any, Dict, List, Optional

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class CountingCalculator(Calculator):
    """
    Wrap an ASE Calculator to count gradient (forces) calls.

    Parameters
    ----------
    calc
        The underlying ASE Calculator instance.
    **kwargs
        Additional keyword arguments passed to super().__init__.
    """

    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calc: Calculator, **kwargs: Any) -> None:
        """
        Initialize the CountingCalculator.

        Attributes
        ----------
        calc
            Underlying ASE calculator.
        grad_calls
            Number of times get_forces() has been called.
        """

        self.calc = calc
        super().__init__(**kwargs)
        if "prefix" in self.__dict__:
            del self.__dict__["prefix"]  # remove shadowed attribute
        self.implemented_properties = self.calc.implemented_properties
        self.grad_calls = 0
    
    def __getattr__(self, name):
        if name == "calc" or "calc" not in self.__dict__:
            raise AttributeError(f"{type(self).__name__} has no attribute {name}")
        return getattr(self.calc, name)

    @property
    def prefix(self):
        # Support both attribute-based and property-based prefix resolution
        return getattr(self.calc, "_prefix", getattr(self.calc, "prefix", None))

    @prefix.setter
    def prefix(self, value):
        if hasattr(self.calc, "_prefix"):
            setattr(self.calc, "_prefix", value)
        else:
            setattr(self.calc, "prefix", value)


    def calculate(self,
                  atoms: Optional[Atoms]=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes
                 ) -> None:
        """
        Perform a calculation of energy and forces, and increment the counter.

        Parameters
        ----------
        atoms
            ASE Atoms object describing the system.
        properties
            List of properties to calculate (e.g., ['energy', 'forces']).
        system_changes
            List of system changes that trigger a full recalculation.
        """
        super().calculate(atoms, properties, system_changes)
        forces = self.calc.get_forces(atoms)
        energy = self.calc.get_potential_energy(atoms)
        self.grad_calls+=1
        print("calc called")
        self.results = {
            'energy': energy,
            'forces': forces,
        }