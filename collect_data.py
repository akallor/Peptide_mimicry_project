#!/usr/bin/env python3
"""
Module for collecting HLA-binding peptides from IEDB and CEDAR databases,
and downloading proteomes from UniProt.
"""

import os
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

class DataCollector:
    def __init__(self):
        self.data_dir = Path("data")
        self.hla_peptides_dir = self.data_dir / "hla_peptides"
        self.proteomes_dir = self.data_dir / "proteomes"
        
        # Create directories if they don't exist
        self.hla_peptides_dir.mkdir(parents=True, exist_ok=True)
        self.proteomes_dir.mkdir(parents=True, exist_ok=True)
        
        # IEDB API endpoint
        self.iedb_api = "http://www.iedb.org/api/v3/mhc_ligand_full"
        
        # UniProt API endpoint
        self.uniprot_api = "https://rest.uniprot.org/uniprotkb/stream"

    def fetch_iedb_peptides(self) -> pd.DataFrame:
        """Fetch human-specific HLA-binding peptides from IEDB database."""
        print("Fetching human-specific peptides from IEDB...")
        
        params = {
            "mhc_class": "class i",
            "response_type": "positive",
            "host_organism": "Homo sapiens",  # Specify human host
            "format": "json"
        }
        
        response = requests.get(self.iedb_api, params=params)
        data = response.json()
        
        # Process the data into a DataFrame
        peptides = []
        for entry in data:
            # Additional checks for human specificity
            host = entry.get('host_organism', '').lower()
            mhc_allele = entry.get('mhc_allele', '').lower()
            
            # Only include if:
            # 1. Host is human (Homo sapiens)
            # 2. MHC allele is HLA (Human Leukocyte Antigen)
            if ('homo sapiens' in host or 'human' in host) and ('hla' in mhc_allele):
                peptides.append({
                    'sequence': entry.get('linear_peptide_seq', ''),
                    'mhc_allele': entry.get('mhc_allele', ''),
                    'source': 'IEDB',
                    'host_organism': entry.get('host_organism', ''),
                    'assay_type': entry.get('assay_type', ''),
                    'measurement_type': entry.get('measurement_type', ''),
                    'measurement_value': entry.get('measurement_value', ''),
                    'measurement_inequality': entry.get('measurement_inequality', '')
                })
        
        df = pd.DataFrame(peptides)
        
        # Remove any duplicates based on sequence and MHC allele
        df = df.drop_duplicates(subset=['sequence', 'mhc_allele'])
        
        # Remove any rows with missing sequences or MHC alleles
        df = df.dropna(subset=['sequence', 'mhc_allele'])
        
        # Save both detailed and simplified versions
        # Detailed version with all metadata
        df.to_csv(self.hla_peptides_dir / "human_iedb_peptides_detailed.csv", index=False)
        
        # Simplified version with just essential columns
        simplified_df = df[['sequence', 'mhc_allele', 'source']]
        simplified_df.to_csv(self.hla_peptides_dir / "human_iedb_peptides.csv", index=False)
        
        print(f"Found {len(df)} unique human-specific peptides")
        print(f"Data saved to:")
        print(f"  - Detailed: {self.hla_peptides_dir}/human_iedb_peptides_detailed.csv")
        print(f"  - Simplified: {self.hla_peptides_dir}/human_iedb_peptides.csv")
        
        return simplified_df

    def download_proteome(self, organism: str) -> None:
        """Download proteome for a specific organism from UniProt."""
        print(f"Downloading proteome for {organism}...")
        
        params = {
            "query": f"organism:{organism}",
            "format": "fasta"
        }
        
        response = requests.get(self.uniprot_api, params=params, stream=True)
        
        if response.status_code == 200:
            filename = self.proteomes_dir / f"{organism.replace(' ', '_')}.fasta"
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded proteome for {organism}")
        else:
            print(f"Failed to download proteome for {organism}")

    def read_pathogen_list(self, file_path: str) -> List[str]:
        """Read list of pathogens from a text file."""
        with open(file_path, 'r') as f:
            # Read lines and remove any empty lines or whitespace
            pathogens = [line.strip() for line in f if line.strip()]
        return pathogens

def main():
    parser = argparse.ArgumentParser(description="Download HLA peptides and pathogen proteomes")
    parser.add_argument('--pathogens', nargs='+', help='List of pathogens to download proteomes for')
    parser.add_argument('--pathogen_file', type=str, help='Path to text file containing pathogen names (one per line)')
    
    args = parser.parse_args()
    
    if not args.pathogens and not args.pathogen_file:
        parser.error("Either --pathogens or --pathogen_file must be provided")
    
    collector = DataCollector()
    
    # Fetch human-specific HLA-binding peptides
    iedb_peptides = collector.fetch_iedb_peptides()
    print(f"Downloaded {len(iedb_peptides)} human-specific peptides from IEDB")
    
    # Get list of pathogens
    pathogens = []
    if args.pathogen_file:
        pathogens = collector.read_pathogen_list(args.pathogen_file)
        print(f"Read {len(pathogens)} pathogens from {args.pathogen_file}")
    else:
        pathogens = args.pathogens
    
    # Download proteomes for specified pathogens
    for pathogen in tqdm(pathogens, desc="Downloading proteomes"):
        collector.download_proteome(pathogen)

if __name__ == "__main__":
    main() 
