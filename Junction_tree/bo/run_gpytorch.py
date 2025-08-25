print('Start calculation')
import torch
import torch.nn as nn
import numpy as np
import gpytorch
import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Descriptors
import networkx as nx
from rdkit.Chem import rdmolops
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
import pickle, gzip, os
from optparse import OptionParser
from jtnn_vae import JTNNVAE
from mol_tree import Vocab
from nnutils import create_var
from model_EOF.predict import predict_HTPB
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ========== Argument Parsing ==========
parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path",default='../data/EMs/vocab.txt')
parser.add_option("-m", "--model", dest="model_path",default='../fast_molvae/vae_fdp_git/model_fdp.iter-30000')
parser.add_option("-o", "--save_dir", dest="save_dir",default='bo_model')
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-r", "--seed", dest="random_seed", default=12)
opts, _ = parser.parse_args()

# ========== Preprocessing ==========
lg = rdkit.RDLogger.logger(); lg.setLevel(rdkit.RDLogger.CRITICAL)
vocab = Vocab([x.strip() for x in open(opts.vocab_path)])
model = JTNNVAE(vocab, 450, 56, 20, 3).cuda()
model.load_state_dict(torch.load(opts.model_path))
model.eval()

X = np.loadtxt('latent_features.txt')
y = -np.loadtxt('targets.txt').reshape((-1, 1))
# SA_scores = np.loadtxt('SA_scores.txt')
SA_scores = np.loadtxt('SA_scores.txt')
logP_values = np.loadtxt('logP_values.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
isp_scores = np.loadtxt('isp_scores.txt')
penalties_scores = np.loadtxt('penalties.txt')
# print(np.std(SA_scores), np.mean(SA_scores),'SA_scores')
print(np.std(cycle_scores), np.mean(cycle_scores),'cycle_scores')
print(np.std(penalties_scores), np.mean(penalties_scores),'penalties_scores')
print(np.std(isp_scores), np.mean(isp_scores),'isp_scores')
# isp_scores = np.loadtxt('isp_scores.txt')
# print('Start calculation')
SA_norm = (SA_scores - SA_scores.mean()) / SA_scores.std()
logP_norm = (logP_values - logP_values.mean()) / logP_values.std()
cycle_norm = (cycle_scores - cycle_scores.mean()) / cycle_scores.std()
penalties_norm = (penalties_scores - penalties_scores.mean()) / penalties_scores.std()

# ========== Data Splitting ==========
n = X.shape[0]
np.random.seed(int(opts.random_seed))
perm = np.random.permutation(n)
X_train, X_test = X[perm[:int(0.9*n)]], X[perm[int(0.9*n):]]
y_train, y_test = y[perm[:int(0.9*n)]], y[perm[int(0.9*n):]]

# ========== GPyTorch Model ==========
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational = gpytorch.variational.VariationalStrategy(
            self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)), learn_inducing_locations=True)
        super().__init__(variational)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class GPRegressionModule(nn.Module):
    def __init__(self, train_x, train_y, M=500):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        Z = train_x[np.random.choice(len(train_x), M)].clone().detach().to(train_x.device)
        self.model = GPModel(Z)
        self.model = self.model.to(train_x.device)  # 🔧 关键修复
        self.likelihood = self.likelihood.to(train_x.device)
    def train_model(self, train_x, train_y, lr=0.01, iters=100):
        self.model.train(); self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, train_y.size(0))

        for i in range(iters):
            print(f"Iter {i+1}/{iters}")
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, test_x):
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))
            return preds.mean.cpu(), preds.variance.cpu()
        

# ========== Feature Extraction ==========
# cycle_scores = []
def extract_cycle_features(smiles_rdkit):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 4:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
        # Aromaticity penalty: If the molecule has no aromatic rings, penalty +1, otherwise 0
    has_aromatic = any(bond.GetIsAromatic() for bond in mol.GetBonds())
    # aromatic_penalties.append(-1.0 if not has_aromatic else 0.0)
    return cycle_length

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
    Add linear penalty for CO balance < -10, the smaller the more penalty
    """
    if co_balance > -10:
        return 0.0
    else:
        return (abs(co_balance + 10)) * 2  # Penalty slope adjustable (e.g. 2 times)

def n_mass_penalty_value(n_mass):
    """
    Penalty for nitrogen mass not in [35, 45]
    """
    if 35 <= n_mass <= 45:
        return 0.0
    elif n_mass < 35:
        return (35 - n_mass) * 1.5  # Penalty slope adjustable
    else:
        return (n_mass - 45) * 1.5
def total_penalties(smiles):
    mol = MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return 100.0  # Extremely high penalty
    co_balance = compute_CO_balance(mol)
    n_mass = compute_N_mass_ratio(mol)
    penalty = co_penalty_value(co_balance) + n_mass_penalty_value(n_mass)
    # total_penalties.append(penalty)
    # print(f"CO balance: {co_balance}, N mass ratio: {n_mass}, Penalty: {penalty}", i)
    return penalty

# ========== EI Sampling Function ==========
def expected_improvement(mu, sigma, best_y, device, eps=1e-9):
    mu = mu.to(device)
    sigma = sigma.to(device)
    best_y = best_y.to(device)

    from torch.distributions.normal import Normal
    normal = Normal(0, 1)
    z = (best_y - mu) / (sigma + eps)
    ei = (best_y - mu) * normal.cdf(z) + sigma * normal.log_prob(z).exp()
    return ei

def sample_next_points(model, train_x, train_y, bounds, n_samples=60):
    device = train_x.device
    # print(train_x.shape[1])
    candidates = np.random.uniform(bounds[0], bounds[1], (1000, 56))
    candidates_tensor = torch.tensor(candidates).float().to(device)

    mu, var = model.predict(candidates_tensor)
    mu = mu.flatten()
    sigma = torch.sqrt(var.flatten() + 1e-9)

    best_y = train_y.min().detach().clone().to(device)

    # 🔧 Call the improved EI
    ei = expected_improvement(mu, sigma, best_y, device)

    top_idx = torch.topk(ei, n_samples).indices.cpu().numpy()
    return candidates[top_idx]

# ========== Main Loop ==========
for run_idx in range(1, 2):
    opts.save_dir = f"result{run_idx}"
    print(opts.save_dir, '********')
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    iteration = 0
    X_train_tensor = torch.tensor(X_train).float().cuda()
    y_train_tensor = torch.tensor(y_train).squeeze().float().cuda()

    while iteration < 100:
        print(f"Iteration {iteration + 1} (run {run_idx})")
        gp_module = GPRegressionModule(X_train_tensor, y_train_tensor)
        gp_module.train_model(X_train_tensor, y_train_tensor, lr=0.01, iters=200)

        bounds = [X_train.min(0), X_train.max(0)]
        next_inputs = sample_next_points(gp_module, X_train_tensor, y_train_tensor, bounds)

        valid_smiles = []
        new_features = []
        for vec in next_inputs:
            vec = vec.reshape(1, -1)
            tree_vec, mol_vec = np.hsplit(vec, 2)
            s = model.decode(create_var(torch.tensor(tree_vec).float().cuda()), create_var(torch.tensor(mol_vec).float().cuda()), prob_decode=False)
            if s:
                valid_smiles.append(s)
                new_features.append(vec)

        print(f"{len(valid_smiles)} molecules are found")
        valid_smiles = valid_smiles[:50]
        new_features = np.vstack(new_features[:50])
        with gzip.open(f"{opts.save_dir}/valid_smiles{iteration}.dat", 'wb') as f:
            f.write(pickle.dumps(valid_smiles))

        scores = []
        for i in range(len(valid_smiles)):
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 4:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 4
            # Aromaticity penalty: If the molecule has no aromatic rings, penalty +1, otherwise 0
            G = nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ])))
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
            current_cycle_score = -cycle_length+chain_penalty
            current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
            # print(current_cycle_score,np.mean(cycle_scores), np.std(cycle_scores), 'cycle_scores')
            # current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[i]))
            current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[i]))
            total_penalties_scores = total_penalties(valid_smiles[i])
            current_isp_scores, ratio_list,EOF_list = predict_HTPB([valid_smiles[i]],[18,30,12,40])
            current_isp_score_normalized = (current_isp_scores[0] - np.mean(isp_scores)) / np.std(isp_scores)
            current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
            # isp_scores_normalized = (current_isp_scores[0] - np.mean(isp_scores)) / np.std(isp_scores)
            current_penalties_normalized = (np.array(total_penalties_scores) - np.mean(penalties_scores)) / np.std(penalties_scores)

            # current_cycle_scores_normalized = (np.array(current_cycle_score) - np.mean(cycle_scores)) / np.std(cycle_scores)
            # score = current_SA_score_normalized+3*current_isp_score_normalized
            # score = current_SA_score_normalized + current_cycle_score_normalized - 0*current_penalties_normalized + current_isp_score_normalized
            # score = current_isp_score_normalized
            score = current_SA_score_normalized + current_cycle_score_normalized - current_penalties_normalized
            # print(f"分子 {i}: SA: {current_SA_score:.2f}, Cycle: {current_cycle_score:.2f}, ISP: {current_isp_scores[0]:.2f}, Penalties: {total_penalties_scores:.2f}, Score: {score:.2f}")
            # print(f"分子 {i}: SA: {current_SA_score_normalized:.2f}, Cycle: {current_cycle_score_normalized:.2f}, ISP: {current_isp_score_normalized:.2f}, Penalties: {total_penalties_scores:.2f}, Score: {score:.2f}")
            # score = current_isp_score_normalized
            # score = - current_penalties_normalized
            print(np.std(SA_scores), np.mean(SA_scores),'SA_scores')
            scores.append(-score)
        print(valid_smiles)
        print(scores)
        with gzip.open(f"{opts.save_dir}/scores{iteration}.dat", 'wb') as f:
            f.write(pickle.dumps(scores))

        if len(new_features) > 0:
            scores_array = np.array(scores).flatten().reshape(-1, 1)
            X_train = np.concatenate([X_train, new_features])
            y_train = np.concatenate([y_train, scores_array])

        X_train_tensor = torch.tensor(X_train).float().cuda()
        y_train_tensor = torch.tensor(y_train).squeeze().float().cuda()
        iteration += 1
