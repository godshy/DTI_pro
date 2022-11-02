
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

EC50 = np.loadtxt('./test_smile', dtype=str)
for m in EC50:
    m = str(m)
mols_bacpi = []

for m in EC50:
    try:
        mols_bacpi.append(Chem.MolFromSmiles(m))
    except TypeError:
        continue
fp_bacpi = []
for mol in mols_bacpi:
    morganfp_bacpi = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    fp_bacpi.append(morganfp_bacpi)
fp_bacpi =np.array(fp_bacpi, dtype='float32')
fp_bacpi.shape
