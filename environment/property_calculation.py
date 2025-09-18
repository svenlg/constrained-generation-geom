import dgl
import torch
from torch_geometric.data import Data

import dxtb
from rdkit import Chem
from dxtb.calculators import GFN1Calculator
from dxtb.components.field import ElectricField

dxtb.timer.disable()

# FlowMol uses a different convention for atom types idx compared to QM9 and GEOM datasets.
# Se: is in there since we sample with CTMC the "non-type" is Se.
# Other FlowMol sampling dont use Se.
atom_type_list = [
    "H", "C", "N", "O", "F", "Se",
]

atom_type_list_qm9 = [
    "C", "H", "N", "O", "F", "Se",
]

atom_type_list_geom = [
    "C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Se",
]

atom_type_to_atomic_number = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53, "Se": 34,
}

bond_type_map = [
    None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, None, # None is used for masked bonds
]
bond_type_to_idx = {bond_type: idx for idx, bond_type in enumerate(bond_type_map)}
bond_type_to_idx[None] = 0

def get_atom_type_map(molecules):
    shape = molecules.ndata['a_t'].shape[1]
    if shape == 6:
        return atom_type_list_qm9
    elif shape == 11:
        return atom_type_list_geom
    else:
        raise ValueError(
            f"Unsupported atom type shape: {shape}. "
            "Expected 6 for QM9 or 11 for GEOM. (Note CTMC uses 6/11 atom types)"
        )

def check_tensor(t):
    if not torch.is_tensor(t):
        raise TypeError("Input is not a tensor")
    if torch.isnan(t).any():
        raise ValueError("Tensor contains NaNs")
    if torch.isinf(t).any():
        raise ValueError("Tensor contains inf or -inf")

def safe_calc(calc_fn, pos, return_grad, device):
    try:
        val = calc_fn(pos)
        if return_grad:
            val_grad = torch.autograd.grad(val, pos)[0]
            # check_tensor(val_grad)
        else:
            val_grad = None
    except (RuntimeError, ValueError) as e:
        print(f"Error in dxtb calculation: {e}", flush=True)
        val = torch.tensor(float("inf"), device=device)
        val_grad = torch.zeros_like(pos, device=device) if return_grad else None
    return val, val_grad

def compute(molecule, atom_type_map, property, reward_lambda, device, return_grad=False):
    """
    Compute the property for a molecule using dxtb.
    """
    # Convert molecule to atomic numbers and positions
    atomic_types_idx = molecule.ndata['a_t'].argmax(dim=1)
    atomic_numbers = [atom_type_to_atomic_number[atom_type_map[atom]] for atom in atomic_types_idx]
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int32, device=device)
    positions = molecule.ndata['x_t']
    positions = positions.clone().detach().requires_grad_(True)

    # Initialize Electric Field
    dd = {"dtype": torch.float32, "device": torch.device(device)}
    field = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32, requires_grad=True)
    ef = ElectricField(field=field, **dd)
    # Initialize dxtb calculator
    opts = {"verbosity": 0, "int_driver": "libcint"}
    calc = GFN1Calculator(atomic_numbers, interaction=ef, opts=opts, **dd)

    if property == "energy":
        def calc_energy(p): return calc.energy(p)
        val, val_grad = safe_calc(calc_energy, positions, return_grad, device)
    elif property == "dipole":
        def dipole_norm(p): return torch.norm(calc.dipole(p))
        val, val_grad = safe_calc(dipole_norm, positions, return_grad, device)
    elif property == "forces":
        def calc_force(p): return torch.norm(calc.forces(p))
        val, val_grad = safe_calc(calc_force, positions, return_grad, device)
    else:
        raise ValueError(f"Unknown property: {property}")

    # Reset calculator
    calc.reset()

    # val = reward_lambda * val.clone().detach()
    val = val.clone().detach()
    if return_grad and val_grad is not None:
        val_grad = reward_lambda * val_grad.clone().detach()

    return val, val_grad

def sa_score(molecule, model, reward_lambda, device, return_grad=False):
    """
    Compute the SAScore gradient for a molecule using a pre-trained model.
    """
    model.eval()
    model.to(device)
    x = molecule.ndata['a_t'].argmax(dim=1)  # shape: [n] 
    # QM9:     C at index 1 and H at index 0 
    # FLowMol: C at index 0 and H at index 1
    # --> swap the indices
    x_tmp = x.clone().detach()
    x[x_tmp == 0] = 1  # H -> C
    x[x_tmp == 1] = 0  # C -> H
    x = x.long().to(device)  # shape: [n]
    pos = molecule.ndata['x_t']  # shape: [n, 3]
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)  # shape: [n]
    src, dst = molecule.edges()
    edge_index = torch.stack([src, dst], dim=0)  # shape: [2, m]
    data = Data(x=x, edge_index=edge_index, pos=pos, batch=batch)
    data = data.to(device)
    if return_grad:
        pos.requires_grad_(True)
    val = model(data)
    if return_grad:
        grad = torch.autograd.grad(val, pos, retain_graph=False)[0]
        grad = reward_lambda * grad.clone().detach()
    else:
        grad = None
    
    return val.clone().detach(), grad

def compute_property_value_and_grad(mol, property, atom_type_map, reward_lambda, device, model, return_grad):
    if property in ["energy", "dipole", "forces"]:
        return compute(
            molecule=mol,
            atom_type_map=atom_type_map,
            property=property,
            reward_lambda=reward_lambda,
            device=device,
            return_grad=return_grad,
        )
    elif property == "const":
        val = torch.tensor(0.0, device=device)
        grad = torch.zeros_like(mol.ndata['x_t'], device=device) if return_grad else None
        return val, grad
    elif property == "sascore":
        assert model is not None, "Model must be provided for SAScore calculation"
        return sa_score(
            molecule=mol,
            model=model,
            reward_lambda=reward_lambda,
            device=device,
            return_grad=return_grad,
        )
    else:
        raise ValueError(f"Unknown property: {property}")


def compute_property_stats(molecules, property, device, model=None):
    properties = []
    atom_type_map = get_atom_type_map(molecules)

    for mol in dgl.unbatch(molecules):
        val, _ = compute_property_value_and_grad(
            mol, property, atom_type_map, reward_lambda=1.0, device=device, model=model, return_grad=False
        )
        properties.append(val)

    properties = torch.stack(properties)
    finite_mask = torch.isfinite(properties)
    properties_finite = properties[finite_mask].detach().cpu()
    return {
        "mean": properties_finite.mean().item(),
        "std": properties_finite.std().item(),
        "max": properties_finite.max().item(),
        "min": properties_finite.min().item(),
        "num_invalid": len(properties) - len(properties_finite),
        "median": properties_finite.median().item(),
        "all_finite": properties_finite,
    }


def compute_property_grad(molecules, property, reward_lambda, device, model=None):
    properties, gradients = [], []
    atom_type_map = get_atom_type_map(molecules)

    for mol in dgl.unbatch(molecules):
        val, grad = compute_property_value_and_grad(
            mol, property, atom_type_map, reward_lambda, device, model, return_grad=True
        )
        properties.append(val)
        gradients.append(grad)

    return properties, gradients
