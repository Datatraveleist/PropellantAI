import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser
from rdkit import Chem
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sascorer
import numpy as np  
from mol_tree import Vocab, MolTree
from jtnn_vae import JTNNVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
from model_EOF.predict import predict_HTPB
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-a", "--data", dest="data_path",default='../data/EMs/valid_smiles.txt')
parser.add_option("-v", "--vocab", dest="vocab_path",default='../data/EMs/vocab.txt')
parser.add_option("-m", "--model", dest="model_path",default='../fast_molvae/vae_fdp/model_fdp.iter-30000')
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
with open(opts.data_path) as f:
    smiles_list = [line.strip() for line in f]
valid_smiles = []
invalid_smiles = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_smiles.append(smi)
    else:
        try:
            MolTree(smi)
            Chem.Kekulize(mol)
            valid_smiles.append(smi)
        except Exception:
            invalid_smiles.append(smi)

print(f"Valid: {len(valid_smiles)}, Invalid: {len(invalid_smiles)}")

# with open(opts.data_path) as f:
#     smiles = f.readlines()
smiles = valid_smiles
for i in range(len(smiles)):
    smiles[ i ] = smiles[ i ].strip()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = 100
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

# model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
model = JTNNVAE(vocab,450, 56, 20, 3)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

smiles_rdkit = []
for i in range(len(smiles)):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ]), isomericSmiles=True))

logP_values = []
for i in range(len(smiles)):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))

SA_scores = []
for i in range(len(smiles)):
    # print(i)
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))

isp_scores = []
for i in range(len(smiles)):
    print(smiles_rdkit[ i ],i)
# isp,radio = predict_HTPB(smiles_rdkit)
isp_scores, ratio_list,EOF_list = predict_HTPB(smiles_rdkit,[18,30,12,40])


import networkx as nx

cycle_scores = []
for i in range(len(smiles)):
    mol = MolFromSmiles(smiles_rdkit[i])
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 4:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 4
    has_aromatic = any(bond.GetIsAromatic() for bond in mol.GetBonds())
    if not has_aromatic:
        cycle_length += 1
    # cycle_scores.append(-cycle_length)
# Calculate the longest chain length limit
    G = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
    max_chain_length = 0
    for node in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, node)
        if lengths:
            max_chain_length = max(max_chain_length, max(lengths.values()))

    max_chain_length_allowed = 10
    if max_chain_length > max_chain_length_allowed:
        chain_penalty = -(max_chain_length - max_chain_length_allowed)
    else:
        chain_penalty = 0
    cycle_scores.append(-cycle_length+chain_penalty)
    print(f" Molecule {i} longest chain length: {max_chain_length}, penalty score: {chain_penalty}, cycle length: {cycle_length}")
# cycle_scores.append(-chain_penalty)


############
def compute_CO_balance(mol):
    mw_total = CalcExactMolWt(mol)
    n_C = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C"])
    n_H = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "H"])
    n_N = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"])
    n_O = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O"])
    n_F = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"])
    n_Cl = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "Cl"])
    n_Br = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "Br"])
    n_I = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "I"])
    if n_C == 0:
        return -100.0  # Avoid division by 0, force failure
    # Halogens consume oxygen
    halogen_penalty = n_F + n_Cl + n_Br + n_I
    effective_O = n_O - halogen_penalty

    OB_CO = 1600 * (effective_O - (n_C + n_H / 2)) / mw_total
    return OB_CO

def compute_N_mass_ratio(mol):
    n_N = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"])
    mw_total = CalcExactMolWt(mol)
    return (n_N * 14.0067) / mw_total * 100 if mw_total > 0 else 0
def co_penalty_value(co_balance):
    """
    For CO balance < -10, apply discrete penalty in steps of 20
    """
    if co_balance > -10:
        return 0.0
    else:
        # Penalty increases by 1 for every 20 units decrease
        penalty_level = int(abs(co_balance + 10) // 20) + 1
        return penalty_level * 1.0  # Each level of penalty is 1, adjustable

def n_mass_penalty_value(n_mass):
    """
    For molecules with N mass not in [35, 45], apply discrete penalty in steps of 20
    """
    if 35 <= n_mass <= 45:
        return 0.0
    elif n_mass < 35:
        penalty_level = int((35 - n_mass) // 20) + 1
        return penalty_level * 1.0
    else:
        penalty_level = int((n_mass - 45) // 20) + 1
        return penalty_level * 1.0
# Apply constraints
total_penalties = []
for i in range(len(smiles)):
    # print(smiles_rdkit[ i ],i)
    mol = MolFromSmiles(smiles_rdkit[ i ])
    mol = Chem.AddHs(mol)
    # print(i)
    if mol is None:
        total_penalties.append(10.0)  # Very high penalty
        continue
    co_balance = compute_CO_balance(mol)
    n_mass = compute_N_mass_ratio(mol)
    penalty = co_penalty_value(co_balance) + n_mass_penalty_value(n_mass)
    total_penalties.append(penalty)
    # print(f"CO balance: {co_balance}, N mass ratio: {n_mass}, Penalty: {penalty}", i)

total_penalties = np.array(total_penalties)
print("latent_points shape:", total_penalties.shape)
penalties_normalized = (total_penalties - np.mean(total_penalties)) / np.std(total_penalties)
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
# isp_scores = np.loadtxt('isp_scores.txt')
isp_scores_normalized = (np.array(isp_scores) - np.mean(isp_scores)) / np.std(isp_scores)

latent_points = []
for i in range(0, len(smiles), batch_size):
    batch = smiles[i:i+batch_size]
    mol_vec,var_vec = model.encode_from_smiles_mean(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

# We store the results
latent_points = np.vstack(latent_points)
np.savetxt('latent_features.txt', latent_points)
targets = SA_scores_normalized - penalties_normalized + cycle_scores_normalized
# targets = SA_scores_normalized + 3*isp_scores_normalized
np.savetxt('targets.txt', targets)
np.savetxt('SA_scores.txt', np.array(SA_scores))
np.savetxt('isp_scores.txt', np.array(isp_scores))
np.savetxt('cycle_scores.txt', np.array(cycle_scores))
# np.savetxt('logP_values.txt', np.array(logP_values))
# np.savetxt('penalties.txt', total_penalties)
