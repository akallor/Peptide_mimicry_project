#!/usr/bin/env python3
"""
Module for generating protein embeddings using ESM-2 language model.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import esm
from torch.utils.data import Dataset, DataLoader

class PeptideDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.data_dir = Path("data")
        self.peptide_dir = self.data_dir / "peptide_libraries"
        self.hla_dir = self.data_dir / "hla_peptides"
        self.output_dir = self.data_dir / "embeddings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ESM-2 model
        print(f"Loading {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_embeddings(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of sequences."""
        dataset = PeptideDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                # Batch tokens
                batch_tokens = self.alphabet.encode(batch)
                batch_tokens = batch_tokens.to(self.device)
                
                # Generate embeddings
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
                
                # Extract per-sequence representations
                batch_embeddings = results["representations"][self.model.num_layers]
                
                # Average pool over sequence length (excluding special tokens)
                batch_embeddings = batch_embeddings[:, 1:, :].mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

    def process_all_peptides(self):
        """Process all peptide files and generate embeddings."""
        # Process HLA peptides
        hla_files = list(self.hla_dir.glob("*.csv"))
        for hla_file in hla_files:
            print(f"Processing {hla_file.name}...")
            df = pd.read_csv(hla_file)
            sequences = df['sequence'].tolist()
            
            embeddings = self.generate_embeddings(sequences)
            
            # Save embeddings
            output_file = self.output_dir / f"{hla_file.stem}_embeddings.npy"
            np.save(output_file, embeddings)
            
            # Save sequence to embedding mapping
            mapping_df = pd.DataFrame({
                'sequence': sequences,
                'embedding_file': str(output_file),
                'embedding_index': range(len(sequences))
            })
            mapping_df.to_csv(self.output_dir / f"{hla_file.stem}_mapping.csv", index=False)

        # Process pathogen peptides
        peptide_files = list(self.peptide_dir.glob("*.csv"))
        for peptide_file in peptide_files:
            print(f"Processing {peptide_file.name}...")
            df = pd.read_csv(peptide_file)
            sequences = df['sequence'].tolist()
            
            embeddings = self.generate_embeddings(sequences)
            
            # Save embeddings
            output_file = self.output_dir / f"{peptide_file.stem}_embeddings.npy"
            np.save(output_file, embeddings)
            
            # Save sequence to embedding mapping
            mapping_df = pd.DataFrame({
                'sequence': sequences,
                'embedding_file': str(output_file),
                'embedding_index': range(len(sequences))
            })
            mapping_df.to_csv(self.output_dir / f"{peptide_file.stem}_mapping.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate protein embeddings using ESM-2")
    parser.add_argument('--model_name', type=str, default="esm2_t33_650M_UR50D",
                      help='Name of the ESM-2 model to use')
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(args.model_name)
    generator.process_all_peptides()

if __name__ == "__main__":
    main() 
