import pandas as pd
from molscore.scoring_functions.SA_Score import sascorer
from rdkit import Chem

import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.CRITICAL)


atom_type_list_geom = [
    "C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Se",
]

def get_sacore(rd_mols: list) -> pd.DataFrame:
    sa_scores = []
    for tmp in rd_mols:
        try:
            Chem.SanitizeMol(tmp.rdkit_mol)
        except:
            sa_scores.append(11.0)
            continue
        try:
            score = sascorer.calculateScore(tmp.rdkit_mol)
            sa_scores.append(score)
        except:
            continue
    return pd.DataFrame({"sascore": sa_scores})

