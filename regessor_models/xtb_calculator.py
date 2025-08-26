import os
import subprocess
import tempfile
import shutil
from typing import Dict
import numpy as np
from rdkit.Chem import GetPeriodicTable


class XTBCalculator:
    """Wrapper for XTB calculations."""
    
    def __init__(self, property_name: str):
        self.property_name = property_name.lower()
        self.temp_dir = None
        
        # Check if XTB is available
        if not shutil.which("xtb"):
            raise RuntimeError("XTB not found in PATH. Please install XTB.")
    
    def calculate_property(self, mol_data: Dict) -> float:
        """Calculate molecular property using XTB."""
        try:
            # Create temporary directory for XTB calculation
            with tempfile.TemporaryDirectory() as temp_dir:
                xyz_file = os.path.join(temp_dir, "molecule.xyz")
                
                # Write XYZ file
                self._write_xyz_file(mol_data, xyz_file)
                
                # Run XTB calculation based on property type
                result = self._run_xtb_calculation(xyz_file, temp_dir)
                
                return result
                
        except Exception as e:
            print(f"XTB calculation failed: {e}")
            return np.nan
    
    def _write_xyz_file(self, mol_data: Dict, xyz_file: str):
        """Write molecular data to XYZ file."""
        atomic_numbers = mol_data['atomic_numbers']
        coords = mol_data['coords']
        
        # Convert atomic numbers to element symbols
        element_symbols = []
        pt = GetPeriodicTable()
        
        for atomic_num in atomic_numbers:
            if atomic_num == 1:
                element_symbols.append('H')
            elif atomic_num == 6:
                element_symbols.append('C')
            elif atomic_num == 7:
                element_symbols.append('N')
            elif atomic_num == 8:
                element_symbols.append('O')
            elif atomic_num == 9:
                element_symbols.append('F')
            elif atomic_num == 15:
                element_symbols.append('P')
            elif atomic_num == 16:
                element_symbols.append('S')
            elif atomic_num == 17:
                element_symbols.append('Cl')
            elif atomic_num == 35:
                element_symbols.append('Br')
            else:
                # Use periodic table for other elements
                element_symbols.append(pt.GetElementSymbol(atomic_num))
        
        with open(xyz_file, 'w') as f:
            f.write(f"{len(atomic_numbers)}\n")
            f.write("Generated molecule\n")
            for symbol, coord in zip(element_symbols, coords):
                f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    
    def _run_xtb_calculation(self, xyz_file: str, temp_dir: str) -> float:
        """Run XTB calculation and extract property."""
        
        if self.property_name in ['energy', 'total_energy']:
            # Single point energy calculation
            cmd = ['xtb', xyz_file, '--sp']
            
        elif self.property_name in ['homo', 'lumo', 'gap', 'homo_lumo_gap']:
            # Electronic structure calculation
            cmd = ['xtb', xyz_file, '--sp', '--etemp', '300']
            
        elif self.property_name in ['dipole', 'dipole_moment']:
            # Dipole moment calculation
            cmd = ['xtb', xyz_file, '--sp', '--dipole']
            
        elif self.property_name in ['polarizability']:
            # Polarizability calculation
            cmd = ['xtb', xyz_file, '--sp', '--polar']
            
        elif self.property_name in ['frequencies', 'freq']:
            # Frequency calculation
            cmd = ['xtb', xyz_file, '--ohess']
            
        else:
            # Default: single point energy
            cmd = ['xtb', xyz_file, '--sp']
        
        try:
            # Run XTB calculation
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"XTB failed with return code {result.returncode}")
            
            # Parse output based on property type
            return self._parse_xtb_output(result.stdout, temp_dir)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("XTB calculation timed out")
    
    def _parse_xtb_output(self, output: str, temp_dir: str) -> float:
        """Parse XTB output to extract the desired property."""
        
        if self.property_name in ['energy', 'total_energy']:
            # Extract total energy in Hartree
            for line in output.split('\n'):
                if 'TOTAL ENERGY' in line or 'total energy' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Eh' in part or 'hartree' in part.lower():
                            return float(parts[i-1])
                        elif part.replace('-', '').replace('.', '').isdigit() and len(part) > 3:
                            return float(part)
        
        elif self.property_name in ['homo']:
            # Extract HOMO energy
            for line in output.split('\n'):
                if 'HOMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            return float(part)
        
        elif self.property_name in ['lumo']:
            # Extract LUMO energy
            for line in output.split('\n'):
                if 'LUMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            return float(part)
        
        elif self.property_name in ['gap', 'homo_lumo_gap']:
            # Extract HOMO-LUMO gap
            homo, lumo = None, None
            for line in output.split('\n'):
                if 'HOMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            homo = float(part)
                if 'LUMO' in line and 'eV' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('-', '').replace('.', '').isdigit():
                            lumo = float(part)
            if homo is not None and lumo is not None:
                return lumo - homo
        
        elif self.property_name in ['dipole', 'dipole_moment']:
            # Extract dipole moment
            for line in output.split('\n'):
                if 'dipole moment' in line.lower() and 'Debye' in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('.', '').isdigit():
                            return float(part)
        
        # If we couldn't parse the property, try to extract any energy value
        for line in output.split('\n'):
            if any(keyword in line.lower() for keyword in ['energy', 'total']):
                parts = line.split()
                for part in parts:
                    try:
                        value = float(part)
                        if abs(value) > 0.1:  # Reasonable energy value
                            return value
                    except ValueError:
                        continue
        
        raise ValueError(f"Could not extract {self.property_name} from XTB output")

