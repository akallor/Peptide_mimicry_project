#!/usr/bin/env python3
"""
2D embedding: ESM sequence similarity + elution rank similarity for peptide network construction.
CPU/GPU/TPU ready (uses torch_xla for TPU if available).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import sys
import gc
import torch

# Try to import fair-esm, provide guidance if not found.
try:
    import esm
except ImportError:
    print("Error: The 'fair-esm' library is required. Please install it with 'pip install fair-esm'.")
    sys.exit(1)

# For TPU support, torch_xla is required.
try:
    import torch_xla.core.xla_model as xm
    _TPU_AVAILABLE = True
except ImportError:
    _TPU_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_peptides(file_path, peptide_col="Peptide", el_col="EL_Rank"):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df = df[[peptide_col, el_col] + [col for col in df.columns if col not in [peptide_col, el_col]]]
    return df

class ESM2DNetwork:
    def __init__(self, model_name, device, embedding_batch_size=16):
        self.model_name = model_name
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.num_layers = None

    def load_model(self):
        if self.model is not None:
            return
        logging.info(f"Loading ESM model: {self.model_name} ...")
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}'. Please ensure the model name is correct.")
            logging.error(f"Original error: {e}")
            sys.exit(1)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_layers = self.model.num_layers
        logging.info(f"Model loaded successfully on {self.device} with {self.num_layers} layers.")

    def get_embeddings(self, peptide_list):
        self.load_model()
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(peptide_list), self.embedding_batch_size), desc="Embedding generation"):
                batch_peptides = peptide_list[i : i + self.embedding_batch_size]
                data = [(f"p{idx}", pep.upper()) for idx, pep in enumerate(batch_peptides)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
                token_representations = results["representations"][self.num_layers]
                for j, peptide in enumerate(batch_peptides):
                    embedding = token_representations[j, 1 : len(peptide) + 1].mean(0)
                    all_embeddings.append(embedding.cpu())
                del batch_tokens, results, token_representations
                gc.collect()
                if hasattr(self.device, 'type') and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif hasattr(self.device, 'type') and self.device.type == 'xla':
                    xm.mark_step()
        return torch.stack(all_embeddings).to(self.device)

def main():
    parser = argparse.ArgumentParser(description="2D peptide network: ESM + elution rank")
    parser.add_argument('--human_file', type=str, required=True, help='Human peptide file (Peptide, EL_Rank)')
    parser.add_argument('--bact_file', type=str, required=True, help='Bacterial peptide file (Peptide, EL_Rank, Organism)')
    parser.add_argument('--esm_threshold', type=float, default=0.8, help='Minimum ESM cosine similarity')
    parser.add_argument('--el_diff', type=float, default=0.1, help='Maximum allowed elution rank difference')
    parser.add_argument('--output', type=str, default='network_2d.tsv', help='Output network file')
    parser.add_argument('--model_name', type=str, default='esm2_t36_3B_UR50D', help='Name of the ESM model to use')
    parser.add_argument('--embedding_batch_size', type=int, default=16, help='Batch size for ESM embedding generation')
    args = parser.parse_args()

    # Device selection
    if _TPU_AVAILABLE:
        logging.info("TPU detected. Using PyTorch/XLA.")
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

    # Load data
    human = load_peptides(args.human_file)
    bact = load_peptides(args.bact_file)
    organism = bact['Organism'].iloc[0] if 'Organism' in bact.columns else "Unknown"

    # Prepare ESM embeddings
    esmnet = ESM2DNetwork(args.model_name, device, args.embedding_batch_size)
    human_peptides = human['Peptide'].tolist()
    bact_peptides = bact['Peptide'].tolist()
    logging.info(f"Generating ESM embeddings for {len(human_peptides)} human and {len(bact_peptides)} bacterial peptides...")
    human_emb = esmnet.get_embeddings(human_peptides)
    bact_emb = esmnet.get_embeddings(bact_peptides)
    # Normalize for cosine similarity
    human_emb = human_emb / human_emb.norm(dim=1, keepdim=True)
    bact_emb = bact_emb / bact_emb.norm(dim=1, keepdim=True)

    # Compute cosine similarity matrix
    logging.info("Computing cosine similarity matrix...")
    sim_matrix = torch.mm(human_emb, bact_emb.T).cpu().numpy()

    # Compute elution rank differences
    human_el = human['EL_Rank'].values.astype(float)
    bact_el = bact['EL_Rank'].values.astype(float)
    el_diff_matrix = np.abs(human_el[:, None] - bact_el[None, :])

    # Filter by thresholds
    mask = (sim_matrix >= args.esm_threshold) & (el_diff_matrix <= args.el_diff)

    # Output network
    with open(args.output, 'w') as f:
        f.write('source\ttarget\tweight\ttype\torganism\n')
        for i, h_row in human.iterrows():
            for j, b_row in bact.iterrows():
                if mask[i, j]:
                    f.write(f"{h_row['Peptide']}\t{b_row['Peptide']}\t{sim_matrix[i, j]:.6f}\thub-spoke\t{organism}\n")

    print(f"Network file written to {args.output}")

if __name__ == "__main__":
    main() 
