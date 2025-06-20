#!/usr/bin/env python3
"""
Script to compute pairwise ESM-based similarities between pathogen and human peptides.
Optimized for GPU or TPU acceleration using PyTorch.
"""

import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import sys
import gc

# Try to import fair-esm, provide guidance if not found.
try:
    import esm
except ImportError:
    print("Error: The 'fair-esm' library is required. Please install it with 'pip install fair-esm'.")
    sys.exit(1)

# For TPU support, torch_xla is required.
# Instructions for installation can be found at: https://github.com/pytorch/xla
try:
    import torch_xla.core.xla_model as xm
    _TPU_AVAILABLE = True
except ImportError:
    _TPU_AVAILABLE = False


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


class PeptideESMSimilarity:
    """
    Calculates peptide similarity using ESM embeddings on CPU, GPU, or TPU.
    """
    def __init__(self, model_name: str, similarity_threshold: float = 0.8, embedding_batch_size: int = 16):
        """
        Initialize the calculator with a specified ESM model.

        Args:
            model_name: Name of the ESM model to load (e.g., 'esm2_t36_3B_UR50D').
            similarity_threshold: Minimum cosine similarity score to consider peptides as similar.
            embedding_batch_size: Batch size for generating embeddings to manage memory.
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_batch_size = embedding_batch_size
        self.device = self._setup_device()
        self.model_name = model_name
        
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.num_layers = None

    def _setup_device(self):
        """Sets up the appropriate device (TPU, GPU, or CPU)."""
        if _TPU_AVAILABLE:
            logging.info("TPU detected. Using PyTorch/XLA.")
            return xm.xla_device()
        
        logging.info("TPU not found. Falling back to GPU/CPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        return device

    def _load_model(self):
        """Loads the ESM model and moves it to the appropriate device."""
        if self.model is not None:
            return
            
        logging.info(f"Loading ESM model: {self.model_name}...")
        logging.info("This may take a while and require significant download and memory.")
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

    def get_embeddings(self, peptide_list: list[str]) -> torch.Tensor:
        """
        Generates embeddings for a list of peptides.

        Args:
            peptide_list: A list of amino acid sequences.

        Returns:
            A torch.Tensor containing the embeddings for all peptides.
        """
        self._load_model()
        
        all_embeddings = []
        logging.info(f"Generating embeddings for {len(peptide_list)} peptides...")
        
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
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'xla':
                    xm.mark_step()


        return torch.stack(all_embeddings).to(self.device)

    def process_peptides(self, pathogen_file: Path, human_file: Path, output_file: Path, pathogen_processing_batch_size: int):
        """
        Processes peptide files, computes similarities, and saves the results.
        """
        logging.info("Loading peptide data...")
        pathogen_df = pd.read_csv(pathogen_file, sep='\t', low_memory=False)
        human_df = pd.read_csv(human_file, sep='\t', low_memory=False)

        pathogen_peptides = pathogen_df['Peptide'].dropna().unique().tolist()
        human_peptides = human_df['Human_peptide'].dropna().unique().tolist()
        
        pathogen_info = pathogen_df.drop_duplicates('Peptide').set_index('Peptide')['Organism'].to_dict()

        logging.info(f"Processing {len(pathogen_peptides)} unique pathogen peptides and {len(human_peptides)} unique human peptides.")
        
        human_embeddings = self.get_embeddings(human_peptides)
        human_embeddings_norm = human_embeddings / human_embeddings.norm(dim=1, keepdim=True)
        del human_embeddings
        gc.collect()

        similar_pairs = []
        logging.info(f"Comparing pathogen peptides against human peptides in batches of {pathogen_processing_batch_size}...")

        for i in tqdm(range(0, len(pathogen_peptides), pathogen_processing_batch_size), desc="Processing pathogen batches"):
            pathogen_batch = pathogen_peptides[i : i + pathogen_processing_batch_size]
            
            if not pathogen_batch:
                continue

            pathogen_batch_embeddings = self.get_embeddings(pathogen_batch)
            pathogen_batch_norm = pathogen_batch_embeddings / pathogen_batch_embeddings.norm(dim=1, keepdim=True)
            
            sim_matrix = torch.mm(pathogen_batch_norm, human_embeddings_norm.T)

            indices = torch.where(sim_matrix >= self.similarity_threshold)
            pathogen_indices, human_indices = indices[0].cpu(), indices[1].cpu()

            for p_idx, h_idx in zip(pathogen_indices, human_indices):
                pathogen_pep = pathogen_batch[p_idx]
                human_pep = human_peptides[h_idx]
                score = sim_matrix[p_idx, h_idx].item()
                
                similar_pairs.append({
                    'pathogen_peptide': pathogen_pep,
                    'human_peptide': human_pep,
                    'similarity_score': score,
                    'pathogen_organism': pathogen_info.get(pathogen_pep, 'N/A'),
                    'pathogen_length': len(pathogen_pep),
                    'human_length': len(human_pep)
                })
            
            del pathogen_batch_embeddings, pathogen_batch_norm, sim_matrix
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'xla':
                xm.mark_step()


        if not similar_pairs:
            print("No similar peptide pairs found with the current threshold.")
            return

        logging.info(f"Found {len(similar_pairs)} similar pairs. Preparing results...")
        results_df = pd.DataFrame(similar_pairs)
        results_df.to_csv(output_file, sep='\t', index=False)

        network_data = []
        for _, row in results_df.iterrows():
            network_data.append({
                'source': row['human_peptide'],
                'target': row['pathogen_peptide'],
                'weight': row['similarity_score'],
                'type': 'hub-spoke',
                'organism': row['pathogen_organism']
            })

        network_file = output_file.parent / f"{output_file.stem}_network.tsv"
        network_df = pd.DataFrame(network_data)
        network_df.to_csv(network_file, sep='\t', index=False)
        
        print(f"\nFound {len(results_df)} similar peptide pairs.")
        print(f"Results saved to:")
        print(f"- Full results: {output_file}")
        print(f"- Network format: {network_file}")

        print("\nSummary by organism:")
        print(results_df.groupby('pathogen_organism').agg({
            'pathogen_peptide': 'count',
            'similarity_score': ['mean', 'min', 'max']
        }).rename(columns={'pathogen_peptide': 'count'}))

        hub_stats = results_df.groupby('human_peptide').size().describe()
        print("\nHub Statistics (pathogenic peptides per human peptide):")
        print(hub_stats)


def main():
    parser = argparse.ArgumentParser(description="Compute ESM-based similarities between pathogen and human peptides.")
    parser.add_argument('--pathogen_file', type=str, required=True, help='Path to pathogen peptides TSV file.')
    parser.add_argument('--human_file', type=str, required=True, help='Path to human peptides TSV file.')
    parser.add_argument('--output', type=str, default='esm_peptide_similarities.tsv', help='Output file path.')
    parser.add_argument('--model_name', type=str, default='esm2_t36_3B_UR50D', help='Name of the ESM model to use.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (e.g., 0.8 for high similarity).')
    parser.add_argument('--embedding_batch_size', type=int, default=16, help='Batch size for generating embeddings (adjust based on GPU memory).')
    parser.add_argument('--processing_batch_size', type=int, default=512, help='Batch size for the main similarity comparison loop.')
    
    args = parser.parse_args()
    
    # Add a log file handler
    file_handler = logging.FileHandler(f'{Path(args.output).stem}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    calculator = PeptideESMSimilarity(
        model_name=args.model_name,
        similarity_threshold=args.threshold,
        embedding_batch_size=args.embedding_batch_size
    )
    calculator.process_peptides(
        Path(args.pathogen_file),
        Path(args.human_file),
        Path(args.output),
        pathogen_processing_batch_size=args.processing_batch_size
    )

if __name__ == "__main__":
    main() 
