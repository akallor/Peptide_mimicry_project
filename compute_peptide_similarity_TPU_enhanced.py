#!/usr/bin/env python3
"""
Script to compute pairwise BLOSUM62 similarities between pathogen and human peptides.
Optimized for TPU acceleration using JAX.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.Align import substitution_matrices
from tqdm import tqdm
import argparse
import logging
import sys
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.lib import xla_bridge
from functools import partial
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('peptide_similarity.log')
    ]
)

class PeptideSimilarityCalculator:
    def __init__(self, similarity_threshold: float = 0.7, batch_size: int = 1024):
        """
        Initialize the calculator with BLOSUM62 matrix and parameters.
        
        Args:
            similarity_threshold: Minimum similarity score to consider peptides as similar
            batch_size: Size of batches for TPU processing
        """
        self.blosum62 = substitution_matrices.load("BLOSUM62")
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.valid_amino_acids = set(self.blosum62.alphabet)
        
        # Convert BLOSUM62 matrix to JAX array
        self.amino_acids = sorted(list(self.valid_amino_acids))
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
        blosum_matrix = np.zeros((len(self.amino_acids), len(self.amino_acids)))
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                blosum_matrix[i, j] = self.blosum62[aa1, aa2]
        
        # Move matrix to TPU
        self.blosum_matrix = jnp.array(blosum_matrix)
        
        # Compile TPU functions
        self._init_tpu_functions()
        
    def _init_tpu_functions(self):
        """Initialize and compile TPU-optimized functions"""
        logging.info(f"Initializing on {xla_bridge.get_backend().platform}")
        
        # Compile single sequence comparison
        @jit
        def _compute_single_score(seq1_indices, seq2_indices, valid_mask):
            """Compute similarity score for single sequence pair"""
            scores = self.blosum_matrix[seq1_indices, seq2_indices]
            masked_scores = scores * valid_mask
            score_sum = jnp.sum(masked_scores)
            max_scores = jnp.maximum(
                self.blosum_matrix[seq1_indices, seq1_indices],
                self.blosum_matrix[seq2_indices, seq2_indices]
            )
            max_sum = jnp.sum(max_scores * valid_mask)
            min_scores = jnp.min(self.blosum_matrix, axis=1)[seq1_indices]
            min_sum = jnp.sum(min_scores * valid_mask)
            
            return (score_sum - min_sum) / (max_sum - min_sum)
        
        # Vectorize over batch dimension
        self._batch_compute = vmap(_compute_single_score, in_axes=(0, 0, 0))
        
        # Parallelize across TPU cores
        self._pmap_compute = pmap(self._batch_compute)
        
    def _encode_sequence(self, seq: str, max_len: int) -> tuple:
        """Convert amino acid sequence to numerical indices with mask"""
        seq = seq.upper()
        indices = np.zeros(max_len, dtype=np.int32)
        mask = np.zeros(max_len, dtype=np.int32)
        
        for i, aa in enumerate(seq[:max_len]):
            if aa in self.aa_to_idx:
                indices[i] = self.aa_to_idx[aa]
                mask[i] = 1
                
        return indices, mask
    
    def _prepare_batch_data(self, seqs1, seqs2):
        """Prepare sequences for batch processing"""
        max_len = max(max(len(s1) for s1 in seqs1), max(len(s2) for s2 in seqs2))
        
        batch_size = len(seqs1)
        seq1_indices = np.zeros((batch_size, max_len), dtype=np.int32)
        seq2_indices = np.zeros((batch_size, max_len), dtype=np.int32)
        valid_masks = np.zeros((batch_size, max_len), dtype=np.int32)
        
        for i, (s1, s2) in enumerate(zip(seqs1, seqs2)):
            s1_idx, s1_mask = self._encode_sequence(s1, max_len)
            s2_idx, s2_mask = self._encode_sequence(s2, max_len)
            seq1_indices[i] = s1_idx
            seq2_indices[i] = s2_idx
            valid_masks[i] = s1_mask * s2_mask
            
        return jnp.array(seq1_indices), jnp.array(seq2_indices), jnp.array(valid_masks)
    
    def compute_batch_similarities(self, pathogen_peptides, human_peptides):
        """Compute similarities for a batch of peptide pairs using TPU"""
        seq1_indices, seq2_indices, valid_masks = self._prepare_batch_data(
            pathogen_peptides, human_peptides
        )
        
        # Reshape for TPU cores if necessary
        num_devices = jax.device_count()
        if len(pathogen_peptides) % num_devices != 0:
            pad_size = num_devices - (len(pathogen_peptides) % num_devices)
            seq1_indices = jnp.pad(seq1_indices, ((0, pad_size), (0, 0)))
            seq2_indices = jnp.pad(seq2_indices, ((0, pad_size), (0, 0)))
            valid_masks = jnp.pad(valid_masks, ((0, pad_size), (0, 0)))
            
        # Reshape for TPU cores
        batch_per_device = len(seq1_indices) // num_devices
        seq1_indices = seq1_indices.reshape((num_devices, batch_per_device, -1))
        seq2_indices = seq2_indices.reshape((num_devices, batch_per_device, -1))
        valid_masks = valid_masks.reshape((num_devices, batch_per_device, -1))
        
        # Compute similarities
        similarities = self._pmap_compute(seq1_indices, seq2_indices, valid_masks)
        
        # Flatten and trim padding if necessary
        similarities = similarities.reshape(-1)[:len(pathogen_peptides)]
        return np.array(similarities)
    
    def process_peptides(self, pathogen_file: Path, human_file: Path, output_file: Path):
        """
        Process peptide files and compute similarities using TPU acceleration.
        """
        # Load peptide data
        print("Loading peptide data...")
        pathogen_df = pd.read_csv(pathogen_file, sep='\t', low_memory=False)
        human_df = pd.read_csv(human_file, sep='\t', low_memory=False)
        
        # Validate input data
        logging.info("Validating peptide sequences...")
        pathogen_peptides = pathogen_df['Peptide'].tolist()
        human_peptides = human_df['Human_peptide'].tolist()
        
        print(f"Processing {len(pathogen_peptides)} pathogen peptides and {len(human_peptides)} human peptides...")
        print(f"Using TPU with {jax.device_count()} cores")
        
        similar_pairs = []
        total_batches = (len(pathogen_peptides) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(pathogen_peptides), self.batch_size), total=total_batches):
            batch_pathogen = pathogen_peptides[i:i + self.batch_size]
            batch_human = human_peptides[:self.batch_size]  # Process against first batch_size human peptides
            
            try:
                similarities = self.compute_batch_similarities(batch_pathogen, batch_human)
                
                # Record similar pairs
                for j, similarity in enumerate(similarities):
                    if similarity >= self.similarity_threshold:
                        similar_pairs.append({
                            'pathogen_peptide': batch_pathogen[j],
                            'human_peptide': batch_human[j],
                            'similarity_score': float(similarity),
                            'pathogen_organism': pathogen_df.iloc[i + j]['Organism'],
                            'pathogen_length': len(batch_pathogen[j]),
                            'human_length': len(batch_human[j])
                        })
            except Exception as e:
                logging.error(f"Error processing batch starting at index {i}: {str(e)}")
                continue
        
        # Create output DataFrame
        if similar_pairs:
            results_df = pd.DataFrame(similar_pairs)
            
            # Save full results
            results_df.to_csv(output_file, sep='\t', index=False)
            
            # Create hub-and-spoke network format
            network_data = []
            for human_pep, group in results_df.groupby('human_peptide'):
                group_sorted = group.sort_values('similarity_score', ascending=False)
                
                for _, row in group_sorted.iterrows():
                    network_data.append({
                        'source': row['human_peptide'],
                        'target': row['pathogen_peptide'],
                        'weight': row['similarity_score'],
                        'type': 'hub-spoke',
                        'organism': row['pathogen_organism']
                    })
            
            # Save network format
            network_file = output_file.parent / f"{output_file.stem}_network.tsv"
            network_df = pd.DataFrame(network_data)
            network_df.to_csv(network_file, sep='\t', index=False)
            
            print(f"\nFound {len(results_df)} similar peptide pairs")
            print(f"Results saved to:")
            print(f"- Full results: {output_file}")
            print(f"- Network format: {network_file}")
            
            # Print summary statistics
            print("\nSummary by organism:")
            print(results_df.groupby('pathogen_organism').agg({
                'pathogen_peptide': 'count',
                'similarity_score': ['mean', 'min', 'max']
            }))
            
            # Print hub statistics
            hub_stats = results_df.groupby('human_peptide').size().describe()
            print("\nHub Statistics (pathogenic peptides per human peptide):")
            print(hub_stats)
        else:
            print("No similar peptide pairs found!")

def main():
    parser = argparse.ArgumentParser(description="Compute BLOSUM62 similarities between pathogen and human peptides")
    parser.add_argument('--pathogen_file', type=str, required=True,
                      help='Path to pathogen peptides TSV file')
    parser.add_argument('--human_file', type=str, required=True,
                      help='Path to human peptides TSV file')
    parser.add_argument('--output', type=str, default='peptide_similarities.tsv',
                      help='Output file path (default: peptide_similarities.tsv)')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Similarity threshold (default: 0.7)')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Batch size for TPU processing (default: 1024)')
    
    args = parser.parse_args()
    
    calculator = PeptideSimilarityCalculator(
        similarity_threshold=args.threshold,
        batch_size=args.batch_size
    )
    calculator.process_peptides(
        Path(args.pathogen_file),
        Path(args.human_file),
        Path(args.output)
    )

if __name__ == "__main__":
    main() 
