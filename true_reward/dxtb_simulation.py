import dgl
import dxtb
import torch

dxtb.timer.disable()

atom_type_list = [
    "H",  # Hydrogen
    "C",  # Carbon
    "N",  # Nitrogen
    "O",  # Oxygen
    "F",  # Fluorine
    "Se",  # Selenium
]

atom_type_to_atomic_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Se": 34,
}

def compute(molecule, reward_lambda, device, return_grad=False):
    """
    Compute the energy for a molecule using dxtb.
    
    Args:
        molecule: The molecule object SampledMolecule (FlowMol output).

    Returns:
        The computed energy.
    """
    # Convert molecule to atomic numbers and positions
    positions = molecule.ndata['x_t']
    atomic_types_idx = molecule.ndata['a_t'].argmax(dim=1)
    atomic_numbers = [atom_type_to_atomic_number[atom_type_list[atom]] for atom in atomic_types_idx]
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int32, device=device)

    # Initialize the dxtb calculator
    dd = {"dtype": torch.float32, "device": torch.device(device)}
    opts = {"verbosity": 0}
    calc = dxtb.calculators.GFN1Calculator(atomic_numbers, opts=opts, **dd)

    pos = positions.clone().requires_grad_(return_grad)
    try:
        energy = - reward_lambda * calc.get_energy(pos)
        if return_grad:
            grad = torch.autograd.grad(energy, pos)[0]
    except RuntimeError as e:
        print(f"Error in dxtb calculation: {e}")
        energy = torch.tensor(float("inf"), device=device)
        if return_grad:
            grad = torch.zeros_like(pos, device=device)
    calc.reset()
    return energy, grad if return_grad else None


def compute_energy(molecules, reward_lambda, device):
    energies = []
    for mol in dgl.unbatch(molecules):
        energy, _ = compute(mol, reward_lambda, device, return_grad=False)
        energies.append(energy)
    energies = torch.stack(energies)
    finite_mask = torch.isfinite(energies)
    energies_finite = energies[finite_mask]
    energies_mean = energies_finite.mean()
    num_invalid = len(energies) - len(energies_finite)
    return energies_mean.detach().cpu().item(), num_invalid


def compute_energy_grad(molecules, reward_lambda, device):
    gradients = []
    for mol in dgl.unbatch(molecules):
        _, grad = compute(mol, reward_lambda, device, return_grad=True)
        gradients.append(grad)
    return gradients

