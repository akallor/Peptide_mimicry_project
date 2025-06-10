#!/usr/bin/env python3
"""
Script to compute pairwise BLOSUM62 similarities between pathogen and human peptides.
Handles peptides of different lengths and produces output suitable for network analysis.
Includes parallel processing for faster computation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.Align import substitution_matrices
from tqdm import tqdm
import argparse
import multiprocessing as mp
from itertools import product
from functools import partial
import os

class PeptideSimilarityCalculator:
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the calculator with BLOSUM62 matrix and parameters.
        
        Args:
            similarity_threshold: Minimum similarity score to consider peptides as similar
        """
        self.blosum62 = substitution_matrices.load("BLOSUM62")
        self.similarity_threshold = similarity_threshold
        
    def compute_normalized_blosum_score(self, seq1: str, seq2: str) -> float:
        """
        Compute normalized BLOSUM62 similarity score between two sequences.
        For sequences of different lengths, uses the length of the shorter sequence
        and slides the shorter sequence along the longer one to find the best match.
        
        Args:
            seq1: First peptide sequence
            seq2: Second peptide sequence
            
        Returns:
            Normalized similarity score between 0 and 1
        """
        # Handle sequences of different lengths
        if len(seq1) == len(seq2):
            return self._compute_single_alignment_score(seq1, seq2)
        else:
            # Use sliding window approach for different lengths
            shorter = seq1 if len(seq1) < len(seq2) else seq2
            longer = seq2 if len(seq1) < len(seq2) else seq1
            
            max_score = float('-inf')
            for i in range(len(longer) - len(shorter) + 1):
                window = longer[i:i + len(shorter)]
                score = self._compute_single_alignment_score(shorter, window)
                max_score = max(max_score, score)
            
            return max_score
    
    def _compute_single_alignment_score(self, seq1: str, seq2: str) -> float:
        """
        Compute normalized BLOSUM62 score for sequences of equal length.
        """
        if not seq1.isalpha() or not seq2.isalpha():
            return 0.0
            
        # Compute raw BLOSUM62 score
        score = 0
        max_possible_score = 0
        min_possible_score = 0
        
        for aa1, aa2 in zip(seq1, seq2):
            try:
                score += self.blosum62[aa1, aa2]
                # Maximum possible score would be the diagonal elements
                max_possible_score += max(self.blosum62[aa1, aa1], self.blosum62[aa2, aa2])
                # Minimum possible score helps in normalization
                min_possible_score += min([self.blosum62[aa1, x] for x in self.blosum62.alphabet])
            except KeyError:
                # Handle non-standard amino acids
                return 0.0
        
        # Normalize score to [0, 1] range
        normalized_score = (score - min_possible_score) / (max_possible_score - min_possible_score)
        return normalized_score

    def _process_batch(self, batch_data):
        """
        Process a batch of peptide pairs in parallel.
        """
        path_idx, path_pep, human_peptides, human_df = batch_data
        similar_pairs = []
        
        for j, hum_pep in enumerate(human_peptides):
            similarity = self.compute_normalized_blosum_score(path_pep['Peptide'], hum_pep)
            
            if similarity >= self.similarity_threshold:
                similar_pairs.append({
                    'pathogen_peptide': path_pep['Peptide'],
                    'human_peptide': hum_pep,
                    'similarity_score': similarity,
                    'pathogen_organism': path_pep['Organism'],
                    'pathogen_length': len(path_pep['Peptide']),
                    'human_length': len(hum_pep)
                })
        
        return similar_pairs

    def process_peptides(self, pathogen_file: Path, human_file: Path, output_file: Path):
        """
        Process peptide files and compute similarities using parallel processing.
        
        Args:
            pathogen_file: Path to pathogen peptides TSV
            human_file: Path to human peptides TSV
            output_file: Path to save similarity results
        """
        # Load peptide data
        print("Loading peptide data...")
        pathogen_df = pd.read_csv(pathogen_file, sep='\t')
        human_df = pd.read_csv(human_file, sep='\t')
        
        # Extract peptide sequences
        pathogen_peptides = pathogen_df.to_dict('records')
        human_peptides = human_df['Name'].tolist()
        
        print(f"Processing {len(pathogen_peptides)} pathogen peptides and {len(human_peptides)} human peptides...")
        
        # Prepare data for parallel processing
        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
        print(f"Using {num_cores} CPU cores for parallel processing")
        
        # Create batches for parallel processing
        batch_data = [(i, path_pep, human_peptides, human_df) 
                     for i, path_pep in enumerate(pathogen_peptides)]
        
        # Process in parallel
        with mp.Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap(self._process_batch, batch_data),
                total=len(batch_data),
                desc="Computing similarities"
            ))
        
        # Flatten results
        similar_pairs = [pair for batch_result in results for pair in batch_result]
        
        # Create output DataFrame
        if similar_pairs:
            results_df = pd.DataFrame(similar_pairs)
            
            # Save full results
            results_df.to_csv(output_file, sep='\t', index=False)
            
            # Create hub-and-spoke network format
            # Group by human peptides to create the hub structure
            network_data = []
            for human_pep, group in results_df.groupby('human_peptide'):
                # Sort pathogen peptides by similarity score
                group_sorted = group.sort_values('similarity_score', ascending=False)
                
                # Add connections from this human peptide (hub) to all similar pathogen peptides
                for _, row in group_sorted.iterrows():
                    network_data.append({
                        'source': row['human_peptide'],  # Hub (human peptide)
                        'target': row['pathogen_peptide'],  # Spoke (pathogen peptide)
                        'weight': row['similarity_score'],
                        'type': 'hub-spoke',  # Indicate this is a hub-spoke connection
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
    
    args = parser.parse_args()
    
    calculator = PeptideSimilarityCalculator(similarity_threshold=args.threshold)
    calculator.process_peptides(
        Path(args.pathogen_file),
        Path(args.human_file),
        Path(args.output)
    )

if __name__ == "__main__":
    main() 
