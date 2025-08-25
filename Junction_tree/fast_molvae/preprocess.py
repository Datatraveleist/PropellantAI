import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle
from mol_tree import Vocab, MolTree
from jtnn_vae import JTNNVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import MolTreeFolder, PairTreeFolder, MolTreeDataset
import rdkit

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    data = []
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path",default='../data/EMs/valid_smiles.txt')
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=20)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    i = 0
    with open(opts.train_path) as f:
        for line in f:
            smile = line.strip("\r\n ").split()[0]
            # tensorize(smile)
            try:
                i = i+1
                # print(i,'/252142')
                tensorize(smile)
                data.append(smile)
            except:
                continue 
    with open("data.txt", "w") as fout:
        for smi in data:
            fout.write(smi + "\n")
    all_data = pool.map(tensorize, data)

    le = (len(all_data) + num_splits - 1) / num_splits
    le = int(le)
    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

