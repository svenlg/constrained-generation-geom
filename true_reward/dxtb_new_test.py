import torch

import dxtb
from dxtb.calculators import GFN1Calculator
from dxtb.components.field import ElectricField

dxtb.timer.disable()

from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# smiles = "CCO"  # Ethanolx
smiles = "C"  # Methane
rdkit_mol = MolFromSmiles(smiles)

# RDKit molecule
molecule = Chem.AddHs(rdkit_mol)  # Add hydrogens if needed
AllChem.EmbedMolecule(molecule)  # Generate 3D coordinates
atomic_numbers = torch.Tensor([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
pos = torch.Tensor(molecule.GetConformer().GetPositions())

atomic_numbers = atomic_numbers.type(torch.int32).to(device)
pos = pos.to(device)
pos = pos.detach().requires_grad_()

dd = {"device": device, "dtype": torch.float32}

field = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32, requires_grad=True)
ef = ElectricField(field=field, **dd)
print(ef)

opts = {"int_driver": "libcint"} # , "verbosity": 0}
calc = GFN1Calculator(atomic_numbers, interaction=ef, opts=opts, **dd)

energy = calc.energy(pos)
energy_grad = torch.autograd.grad(energy, pos)[0]
print("Energy:", energy.item())
print("Energy gradient:", energy_grad)
print()
calc.reset()

dipole = calc.dipole(pos)
dipole_norm = torch.norm(dipole)
dipole_norm_grad = torch.autograd.grad(dipole_norm, pos)[0]
print("Dipole:", dipole)
print("Dipole norm:", dipole_norm.item())
print("Dipole norm gradient:", dipole_norm_grad)
print()
calc.reset()

tmp_lambda = torch.ones_like(dipole_norm_grad, device=dipole_norm_grad.device) * 1.0
tmp_rho = torch.ones_like(dipole_norm_grad, device=dipole_norm_grad.device) * 0.5

full_grad = energy_grad - tmp_lambda * dipole_norm_grad - tmp_rho * dipole_norm_grad

energy_calc = GFN1Calculator(atomic_numbers, interaction=ef, opts=opts, **dd)
dipole_calc = GFN1Calculator(atomic_numbers, interaction=ef, opts=opts, **dd)
energy = energy_calc.energy(pos)
dipole = dipole_calc.dipole(pos)
print("Energy:", energy.item())
print("Dipole:", dipole)
print()



# a = dipole
# b = field
# aflat = a.reshape(-1)
# anumel, bnumel = a.numel(), b.numel()

# res = torch.empty(
#     (anumel, bnumel),
#     dtype=a.dtype,
#     device=a.device,
# )
# create_graph = torch.is_grad_enabled()
# retain_graph = True

# for i in range(aflat.numel()):
#     (g,) = torch.autograd.grad(
#         aflat[i],
#         b,
#         allow_unused=True,
#         retain_graph=True,
#         create_graph=True,
#     )
#     if g is None:
#         print(f"aflat[{i}] does not depend on field!")
#     else:
#         res[i] = g.reshape(-1)


# pol = calc.polarizability(pos, derived_quantity='energy')
# print("Polarizability:", pol)
# print("Polarizability:", pol.shape)

