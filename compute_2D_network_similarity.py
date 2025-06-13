#!/usr/bin/env python3
"""
2D embedding: BLOSUM62 sequence similarity + elution rank similarity for peptide network construction.
CPU/GPU/TPU ready (uses JAX for acceleration if available).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.Align import substitution_matrices
from tqdm import tqdm
import argparse
import jax
import jax.numpy as jnp
from jax import jit, vmap
import os

def load_peptides(file_path, peptide_col="Peptide", el_col="EL_Rank"):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df = df[[peptide_col, el_col] + [col for col in df.columns if col not in [peptide_col, el_col]]]
    return df

def create_blosum_jax():
    blosum62 = substitution_matrices.load("BLOSUM62")
    aa = sorted(list(blosum62.alphabet))
    aa_to_idx = {a: i for i, a in enumerate(aa)}
    mat = np.zeros((len(aa), len(aa)))
    for i, a1 in enumerate(aa):
        for j, a2 in enumerate(aa):
            mat[i, j] = blosum62[a1, a2]
    return jnp.array(mat), aa_to_idx

def seq_to_idx(seq, aa_to_idx, max_len):
    idx = np.zeros(max_len, dtype=np.int32)
    for i, a in enumerate(seq):
        idx[i] = aa_to_idx.get(a, 0)
    return idx

def main():
    parser = argparse.ArgumentParser(description="2D peptide network: BLOSUM + elution rank")
    parser.add_argument('--human_file', type=str, required=True, help='Human peptide file (Peptide, EL_Rank)')
    parser.add_argument('--bact_file', type=str, required=True, help='Bacterial peptide file (Peptide, EL_Rank, Organism)')
    parser.add_argument('--blosum_threshold', type=float, default=0.8, help='Minimum BLOSUM similarity')
    parser.add_argument('--el_diff', type=float, default=0.1, help='Maximum allowed elution rank difference')
    parser.add_argument('--output', type=str, default='network_2d.tsv', help='Output network file')
    args = parser.parse_args()

    # Load data
    human = load_peptides(args.human_file)
    bact = load_peptides(args.bact_file)
    organism = bact['Organism'].iloc[0] if 'Organism' in bact.columns else "Unknown"

    # Prepare BLOSUM
    blosum, aa_to_idx = create_blosum_jax()
    max_len = max(human['Peptide'].str.len().max(), bact['Peptide'].str.len().max())

    # Convert sequences to indices
    human_idx = np.stack([seq_to_idx(seq, aa_to_idx, max_len) for seq in human['Peptide']])
    bact_idx = np.stack([seq_to_idx(seq, aa_to_idx, max_len) for seq in bact['Peptide']])

    # JAX BLOSUM similarity function
    @jit
    def blosum_sim(idx1, idx2):
        scores = blosum[idx1, idx2]
        # Only consider up to the length of the shorter peptide
        valid = (idx1 != 0) & (idx2 != 0)
        scores = scores * valid
        score_sum = jnp.sum(scores)
        max_score = jnp.sum(jnp.maximum(blosum[idx1, idx1], blosum[idx2, idx2]) * valid)
        min_score = jnp.sum(jnp.min(blosum, axis=1)[idx1] * valid)
        return (score_sum - min_score) / (max_score - min_score + 1e-8)

    v_blosum_sim = vmap(lambda h: vmap(lambda b: blosum_sim(h, b))(bact_idx))(human_idx)

    # Compute elution rank differences
    human_el = human['EL_Rank'].values.astype(float)
    bact_el = bact['EL_Rank'].values.astype(float)
    el_diff_matrix = np.abs(human_el[:, None] - bact_el[None, :])

    # Filter by thresholds
    mask = (v_blosum_sim >= args.blosum_threshold) & (el_diff_matrix <= args.el_diff)

    # Output network
    with open(args.output, 'w') as f:
        f.write('source\ttarget\tweight\ttype\torganism\n')
        for i, h_row in human.iterrows():
            for j, b_row in bact.iterrows():
                if mask[i, j]:
                    f.write(f"{h_row['Peptide']}\t{b_row['Peptide']}\t{v_blosum_sim[i, j]:.6f}\thub-spoke\t{organism}\n")

    print(f"Network file written to {args.output}")

if __name__ == "__main__":
    main() 
