import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import dxtb


def molecule_to_xyz(molecule, format_type, device):
    """
    Convert a molecule from RDKit, DGL, or PyTorch Geometric format to atomic numbers and positions.
    
    Args:
        molecule: The molecule object (RDKit Mol, DGLGraph, or PyTorch Geometric Data).
        format_type: The format of the molecule ("rdkit", "dgl", or "pyg" (pytorch_geometric)).
    
    Returns:
        atomic_numbers: List of atomic numbers.
        positions: Array of atomic positions.
    """
    if format_type == "rdkit":
        # RDKit molecule
        molecule = Chem.AddHs(molecule)  # Add hydrogens if needed
        AllChem.EmbedMolecule(molecule)  # Generate 3D coordinates
        # AllChem.MMFFOptimizeMolecule(molecule)  # Optional: Optimize geometry
        atomic_numbers = torch.Tensor([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        positions = torch.Tensor(molecule.GetConformer().GetPositions())
    elif format_type == "dgl":
        # DGL graph
        atomic_numbers = molecule.ndata["atomic_number"]
        positions = molecule.ndata["pos"]
    elif format_type == "pyg":
        # PyTorch Geometric Data object
        atomic_numbers = molecule.z
        positions = molecule.pos
    else:
        raise ValueError("Unsupported format type. Use 'rdkit', 'dgl', or  'pyg' (pytorch_geometric).")

    atomic_numbers = atomic_numbers.type(torch.int32).to(device)
    positions = positions.to(device)
    return atomic_numbers, positions


def compute_quantity_dxtb(atomic_numbers, positions, device="cpu"):
    pos = positions.clone().requires_grad_(True)

    # # Initialize the dxtb calculator
    # dd = {"dtype": torch.float32, "device": torch.device(device)}
    # calc = dxtb.calculators.GFN1Calculator(atomic_numbers, **dd)
    # energy = calc.energy(pos)
    # print(energy)

    # Initialize the dxtb calculator with electric field
    dd = {"dtype": torch.float32, "device": torch.device(device)}
    efield = torch.tensor([1.0, 1.0, 1.0], device=device)
    electric_field = dxtb.components.field.new_efield(efield)
    calc = dxtb.calculators.GFN1Calculator(atomic_numbers, interaction=electric_field, **dd)
    dipole_numerical = calc.dipole(pos)
    print(dipole_numerical)
    dipole_numerical = calc.dipole_numerical(pos)
    print(dipole_numerical)
    assert False

    moin = calc.calculate(["energy", "dipole"],pos)
    print(moin)
    assert False
    calc.get
    # # Now you can calculate the dipole moment
    # dipole = calc.get_dipole_moment(pos, chrg=0, spin=0)
    # # dipole = calc.dipole(pos)

    # # Compute the specified quantity
    # if False:
    #     if quantity == "homolumo":
    #         results = calc.compute("electronic_structure")
    #         homo = results["homo_energy"]
    #         lumo = results["lumo_energy"]
    #         return (lumo - homo).item()  # HOMO-LUMO gap
    #     elif quantity == "energy":
    #         results = calc.compute("energy")
    #         return results["total_energy"].item()

    return energy# , polarizability


def compute_quantity(molecule, format_type, device="cpu"):
    """
    Compute a specified quantity for a molecule using dxtb.
    
    Args:
        molecule: The molecule object (RDKit Mol, DGLGraph, or PyTorch Geometric Data).
        format_type: The format of the molecule ("rdkit", "dgl", or  "pyg" (pytorch_geometric)).
        device: 
    
    Returns:
        The computed quantity.
    """
    # Convert molecule to atomic numbers and positions
    atomic_numbers, positions = molecule_to_xyz(molecule, format_type, device)

    # Compute the quantity using dxtb
    return compute_quantity_dxtb(atomic_numbers, positions, device)


# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example: RDKit molecule
    from rdkit.Chem import MolFromSmiles
    smiles = "CCO"  # Ethanolx
    rdkit_mol = MolFromSmiles(smiles)
    # energy, polarizability = compute_quantity(rdkit_mol, "rdkit", device=device)
    energy = compute_quantity(rdkit_mol, "rdkit", device=device)
    print(f"Energy of the system {energy} Hartree.")
    # print(f"Polarizability {polarizability}")

    # Example: PyTorch Geometric molecule
    from torch_geometric.data import Data
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1])  # Methane
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    pytorch_geo_mol = Data(z=atomic_numbers, pos=positions)
    energy = compute_quantity(pytorch_geo_mol, "pyg", device=device)
    print(f"Energy of the system {energy} Hartree.")