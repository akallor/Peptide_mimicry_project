#!/usr/bin/env python3
"""
ESM-based peptide similarity computation, optimized for GPU (A100, L4, T4) and Colab.
- Uses DataLoader for efficient batching
- Mixed precision (float16) for memory efficiency
- Robust logging and error handling
- No TPU code, only CUDA/CPU

Usage (in Colab or CLI):
!python esm_gpu_peptide_similarity.py --pathogen_file pathogen.tsv --human_file human.tsv --output results.tsv
"""

import os
import sys
import gc
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Try to import fair-esm
try:
    import esm
except ImportError:
    print("Error: The 'fair-esm' library is required. Please install it with 'pip install fair-esm'.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# For Colab: check GPU
if not torch.cuda.is_available():
    logging.warning("CUDA GPU not available. Running on CPU will be very slow!")
else:
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

# --- Data utilities ---
class PeptideDataset(Dataset):
    def __init__(self, peptides: List[str]):
        self.peptides = peptides
    def __len__(self):
        return len(self.peptides)
    def __getitem__(self, idx):
        return self.peptides[idx]

# --- Peptide sequence validation ---
def is_valid_peptide(seq: str) -> bool:
    valid_aas = set("ACDEFGHIKLMNPQRSTVWYUBZOXJ")
    seq = seq.strip().upper()
    return all(aa in valid_aas for aa in seq) and len(seq) > 0

# --- ESM Similarity Calculator ---
class PeptideESMSimilarityGPU:
    def __init__(self, model_name: str, threshold: float = 0.8, batch_size: int = 32, dtype: torch.dtype = torch.float16):
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if self.device.type == 'cuda' else torch.float32
        self._load_model()

    def _load_model(self):
        logging.info(f"Loading ESM model: {self.model_name} ...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_layers = self.model.num_layers
        logging.info(f"Model loaded on {self.device} with {self.num_layers} layers.")

    def get_embeddings(self, peptides: List[str]) -> torch.Tensor:
        # Filter and warn about invalid peptides
        valid_peptides = [p for p in peptides if is_valid_peptide(p)]
        skipped = len(peptides) - len(valid_peptides)
        if skipped > 0:
            logging.warning(f"Skipped {skipped} invalid peptide sequences.")
        if not valid_peptides:
            raise ValueError("No valid peptides provided.")
        dataset = PeptideDataset(valid_peptides)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        all_embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
            for batch in tqdm(loader, desc="Embedding generation", leave=True):
                data = [(f"p{i}", seq.upper()) for i, seq in enumerate(batch)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
                token_representations = results["representations"][self.num_layers]
                for j, seq in enumerate(batch):
                    emb = token_representations[j, 1:len(seq)+1].mean(0)
                    all_embeddings.append(emb.cpu())
                del batch_tokens, results, token_representations
                torch.cuda.empty_cache()
                gc.collect()
        return torch.stack(all_embeddings).to(self.device, dtype=self.dtype)

    def compute_similarities(self, pathogen_emb: torch.Tensor, human_emb: torch.Tensor, pathogen_peptides: List[str], human_peptides: List[str], pathogen_info: dict) -> pd.DataFrame:
        # Normalize
        pathogen_norm = pathogen_emb / pathogen_emb.norm(dim=1, keepdim=True)
        human_norm = human_emb / human_emb.norm(dim=1, keepdim=True)
        # Compute similarity matrix in blocks for memory efficiency
        results = []
        for i in tqdm(range(0, len(pathogen_norm), self.batch_size), desc="Similarity blocks"):
            p_block = pathogen_norm[i:i+self.batch_size]
            sim_block = torch.mm(p_block, human_norm.T)
            idxs = torch.where(sim_block >= self.threshold)
            for pi, hi in zip(idxs[0], idxs[1]):
                p_idx = i + pi.item()
                h_idx = hi.item()
                results.append({
                    'pathogen_peptide': pathogen_peptides[p_idx],
                    'human_peptide': human_peptides[h_idx],
                    'similarity_score': sim_block[pi, hi].item(),
                    'pathogen_organism': pathogen_info.get(pathogen_peptides[p_idx], 'N/A'),
                    'pathogen_length': len(pathogen_peptides[p_idx]),
                    'human_length': len(human_peptides[h_idx])
                })
            del p_block, sim_block, idxs
            torch.cuda.empty_cache()
            gc.collect()
        return pd.DataFrame(results)

# --- Main pipeline ---
def main():
    parser = argparse.ArgumentParser(description="Compute ESM-based similarities between pathogen and human peptides (GPU-optimized).")
    parser.add_argument('--pathogen_file', type=str, required=True, help='Path to pathogen peptides TSV file.')
    parser.add_argument('--human_file', type=str, required=True, help='Path to human peptides TSV file.')
    parser.add_argument('--output', type=str, default='esm_peptide_similarities.tsv', help='Output file path.')
    parser.add_argument('--model_name', type=str, default='esm2_t36_3B_UR50D', help='Name of the ESM model to use.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (e.g., 0.8 for high similarity).')
    parser.add_argument('--embedding_batch_size', type=int, default=32, help='Batch size for generating embeddings (adjust for GPU memory).')
    args = parser.parse_args()

    # Add a log file handler
    file_handler = logging.FileHandler(f'{Path(args.output).stem}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Load data
    logging.info("Loading peptide data...")
    pathogen_df = pd.read_csv(args.pathogen_file, sep='\t', low_memory=False)
    human_df = pd.read_csv(args.human_file, sep='\t', low_memory=False)
    pathogen_peptides = pathogen_df['Peptide'].dropna().unique().tolist()
    human_peptides = human_df['Human_peptide'].dropna().unique().tolist()
    pathogen_info = pathogen_df.drop_duplicates('Peptide').set_index('Peptide')['Organism'].to_dict()
    logging.info(f"{len(pathogen_peptides)} unique pathogen peptides, {len(human_peptides)} unique human peptides.")

    # Initialize calculator
    calculator = PeptideESMSimilarityGPU(
        model_name=args.model_name,
        threshold=args.threshold,
        batch_size=args.embedding_batch_size,
        dtype=torch.float16
    )

    # Embeddings
    logging.info("Generating human peptide embeddings...")
    human_emb = calculator.get_embeddings(human_peptides)
    logging.info("Generating pathogen peptide embeddings...")
    pathogen_emb = calculator.get_embeddings(pathogen_peptides)

    # Similarity
    logging.info("Computing similarities...")
    results_df = calculator.compute_similarities(pathogen_emb, human_emb, pathogen_peptides, human_peptides, pathogen_info)
    results_df.to_csv(args.output, sep='\t', index=False)
    logging.info(f"Results saved to {args.output}")

    # Network format
    network_data = [
        {
            'source': row['human_peptide'],
            'target': row['pathogen_peptide'],
            'weight': row['similarity_score'],
            'type': 'hub-spoke',
            'organism': row['pathogen_organism']
        }
        for _, row in results_df.iterrows()
    ]
    network_file = Path(args.output).parent / f"{Path(args.output).stem}_network.tsv"
    pd.DataFrame(network_data).to_csv(network_file, sep='\t', index=False)
    logging.info(f"Network file saved to {network_file}")

    # Print summary
    print(f"\nFound {len(results_df)} similar peptide pairs.")
    print(f"Results saved to: {args.output}\nNetwork format: {network_file}")
    print("\nSummary by organism:")
    print(results_df.groupby('pathogen_organism').agg({
        'pathogen_peptide': 'count',
        'similarity_score': ['mean', 'min', 'max']
    }).rename(columns={'pathogen_peptide': 'count'}))
    hub_stats = results_df.groupby('human_peptide').size().describe()
    print("\nHub Statistics (pathogenic peptides per human peptide):")
    print(hub_stats)

if __name__ == "__main__":
    main() 
