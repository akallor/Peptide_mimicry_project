#!/usr/bin/env python3
"""
Module for generating peptide libraries from proteomes using sliding windows.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from typing import List, Generator
from tqdm import tqdm

class PeptideLibraryGenerator:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data_dir = Path("data")
        self.proteomes_dir = self.data_dir / "proteomes"
        self.output_dir = self.data_dir / "peptide_libraries"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_peptides(self, sequence: str) -> Generator[str, None, None]:
        """Generate peptides using sliding window approach."""
        for i in range(len(sequence) - self.window_size + 1):
            yield sequence[i:i + self.window_size]

    def process_fasta(self, fasta_file: Path) -> List[str]:
        """Process a FASTA file and generate peptides."""
        peptides = []
        
        for record in SeqIO.parse(str(fasta_file), "fasta"):
            sequence = str(record.seq)
            peptides.extend(self.generate_peptides(sequence))
            
        return peptides

    def generate_library(self) -> None:
        """Generate peptide libraries for all proteomes."""
        # Process each FASTA file in the proteomes directory
        for fasta_file in tqdm(list(self.proteomes_dir.glob("*.fasta")), desc="Processing proteomes"):
            organism_name = fasta_file.stem
            
            print(f"Processing {organism_name}...")
            peptides = self.process_fasta(fasta_file)
            
            # Save peptides to CSV
            df = pd.DataFrame(peptides, columns=['sequence'])
            df['source_organism'] = organism_name
            output_file = self.output_dir / f"{organism_name}_peptides.csv"
            df.to_csv(output_file, index=False)
            
            print(f"Generated {len(peptides)} peptides for {organism_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate peptide libraries using sliding windows")
    parser.add_argument('--window_size', type=int, required=True,
                      help='Size of the sliding window (number of amino acids)')
    
    args = parser.parse_args()
    
    generator = PeptideLibraryGenerator(args.window_size)
    generator.generate_library()

if __name__ == "__main__":
    main() 
