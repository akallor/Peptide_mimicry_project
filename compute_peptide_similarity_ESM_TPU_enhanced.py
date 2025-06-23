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
    @staticmethod
    def is_valid_peptide(seq: str) -> bool:
        """
        Checks if a peptide sequence contains only valid amino acid codes.
        """
        valid_aas = set("ACDEFGHIKLMNPQRSTVWYUBZOXJ")  # standard + rare/ambiguous
        seq = seq.strip().upper()
        return all(aa in valid_aas for aa in seq) and len(seq) > 0

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
        
        # Filter invalid peptides
        valid_peptides = [pep for pep in peptide_list if self.is_valid_peptide(pep)]
        skipped = len(peptide_list) - len(valid_peptides)
        if skipped > 0:
            logging.warning(f"Skipped {skipped} invalid peptide sequences during embedding generation.")
        peptide_list = valid_peptides
        if not peptide_list:
            raise ValueError("No valid peptides provided for embedding generation.")
        
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

        # Filter invalid peptides and log
        pathogen_peptides = pathogen_df['Peptide'].dropna().unique().tolist()
        valid_pathogen_peptides = [pep for pep in pathogen_peptides if self.is_valid_peptide(pep)]
        skipped_pathogen = len(pathogen_peptides) - len(valid_pathogen_peptides)
        if skipped_pathogen > 0:
            logging.warning(f"Skipped {skipped_pathogen} invalid pathogen peptides.")
        pathogen_peptides = valid_pathogen_peptides

        human_peptides = human_df['Human_peptide'].dropna().unique().tolist()
        valid_human_peptides = [pep for pep in human_peptides if self.is_valid_peptide(pep)]
        skipped_human = len(human_peptides) - len(valid_human_peptides)
        if skipped_human > 0:
            logging.warning(f"Skipped {skipped_human} invalid human peptides.")
        human_peptides = valid_human_peptides

        pathogen_info = pathogen_df.drop_duplicates('Peptide').set_index('Peptide')['Organism'].to_dict()

        logging.info(f"Processing {len(pathogen_peptides)} unique pathogen peptides and {len(human_peptides)} unique human peptides.")
        # ... existing code ... 
