#!/usr/bin/env python3
"""
Script to generate peptide libraries from FASTA files using sliding windows.
Takes multiple input FASTA files and their corresponding organism names,
and generates peptides of specified lengths using sliding windows.
"""

import argparse
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from typing import List, Dict, Generator
from tqdm import tqdm

def generate_peptides(sequence: str, min_length: int, max_length: int) -> Generator[tuple, None, None]:
    """
    Generate peptides from a sequence using sliding windows of varying lengths.
    
    Args:
        sequence: Input protein sequence
        min_length: Minimum peptide length
        max_length: Maximum peptide length
    
    Yields:
        Tuple of (peptide sequence, length)
    """
    # Generate peptides for each window size
    for window_size in range(min_length, max_length + 1):
        for i in range(len(sequence) - window_size + 1):
            peptide = sequence[i:i + window_size]
            # Skip peptides containing non-standard amino acidsgene
            if not peptide.isalpha():
                continue
            yield (peptide, window_size)

def process_fasta(fasta_file: Path, organism: str, min_length: int, max_length: int) -> List[Dict]:
    """
    Process a FASTA file and generate peptides.
    
    Args:
        fasta_file: Path to input FASTA file
        organism: Name of the organism
        min_length: Minimum peptide length
        max_length: Maximum peptide length
    
    Returns:
        List of dictionaries containing peptide information
    """
    peptides = []
    
    print(f"\nProcessing {fasta_file.name} for organism: {organism}")
    
    # Process each sequence in the FASTA file
    for record in tqdm(list(SeqIO.parse(str(fasta_file), "fasta")), desc="Processing sequences"):
        sequence = str(record.seq)
        
        # Generate peptides using sliding windows
        for peptide, length in generate_peptides(sequence, min_length, max_length):
            peptides.append({
                'Peptide': peptide,
                'Organism': organism,
                'Length': length
            })
    
    print(f"Generated {len(peptides)} peptides for {organism}")
    return peptides

def main():
    parser = argparse.ArgumentParser(description="Generate peptide library from FASTA files using sliding windows")
    parser.add_argument('--input_pairs', nargs='+', required=True,
                      help='Pairs of FASTA file paths and organism names in format: path1 organism1 path2 organism2 ...')
    parser.add_argument('--min_length', type=int, default=8,
                      help='Minimum peptide length (default: 8)')
    parser.add_argument('--max_length', type=int, default=12,
                      help='Maximum peptide length (default: 12)')
    parser.add_argument('--output', type=str, default='pathogen_peptide_library.tsv',
                      help='Output TSV file path (default: pathogen_peptide_library.tsv)')
    
    args = parser.parse_args()
    
    # Validate input pairs
    if len(args.input_pairs) % 2 != 0:
        parser.error("Input pairs must be in pairs of FASTA path and organism name")
    
    # Process input pairs
    all_peptides = []
    for i in range(0, len(args.input_pairs), 2):
        fasta_path = Path(args.input_pairs[i])
        organism = args.input_pairs[i + 1]
        
        if not fasta_path.exists():
            print(f"Warning: FASTA file not found: {fasta_path}")
            continue
            
        peptides = process_fasta(
            fasta_path,
            organism,
            args.min_length,
            args.max_length
        )
        all_peptides.extend(peptides)
    
    # Create DataFrame and save to TSV
    if all_peptides:
        df = pd.DataFrame(all_peptides)
        df.to_csv(args.output, sep='\t', index=False)
        print(f"\nSaved {len(df)} peptides to {args.output}")
        print(f"Summary by organism:")
        print(df.groupby('Organism').agg({'Peptide': 'count'}).rename(columns={'Peptide': 'Count'}))
    else:
        print("No peptides were generated!")

if __name__ == "__main__":
    main() 
