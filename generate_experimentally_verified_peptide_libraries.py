#Testing the experimentally verified peptide mimicking partners from the miPEP database
import os
import glob
import pandas as pd
from Bio import SeqIO

# --- CONFIG ---
HUMAN_FASTA = "/content/drive/MyDrive/Peptide_mimicry_project/human_data/human_proteome/uniprotkb_Homo_sapiens_AND_reviewed_tru_2025_06_25.fasta"
PATHOGEN_FASTA_GLOB = "/content/drive/MyDrive/Peptide_mimicry_project/pathogen_data/pathogen_proteomes/*.fasta"
TSV_FILE = "/content/drive/MyDrive/Peptide_mimicry_project/mipepbase_table_2.tsv"
HUMAN_OUT = "/content/drive/MyDrive/Peptide_mimicry_project/human_verified_peptide_library.tsv"
PATHOGEN_OUT = "/content/drive/MyDrive/Peptide_mimicry_project/pathogen_verified_peptide_library.tsv"
SELECTED_PATHOGENS = ["Helicobacter pylori"]  # Default: only process this pathogen

# --- 1. Read the table and extract relevant protein names ---
df = pd.read_csv(TSV_FILE, sep="\t")

# Always get all human proteins from the full table
human_proteins = set(df[df["Host"] == "Human"]["Host Protein"].str.strip())

# For pathogens, filter if needed
if SELECTED_PATHOGENS:
    pathogen_df = df[df["Pathogen"].isin(SELECTED_PATHOGENS)]
else:
    pathogen_df = df

pathogen_proteins = set(pathogen_df["Pathogen Protein"].str.strip())
pathogen_organism_map = dict(zip(pathogen_df["Pathogen Protein"].str.strip(), pathogen_df["Pathogen"].str.strip()))

# --- 2. Parse human proteome and extract matching proteins ---
def extract_matching_proteins_from_fasta(fasta_path, protein_names):
    matches = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        # Try to match by exact name in header
        for pname in protein_names:
            if pname in header:
                matches[pname] = str(record.seq)
    return matches

human_matches = extract_matching_proteins_from_fasta(HUMAN_FASTA, human_proteins)

# --- 3. Parse pathogen proteomes and extract matching proteins ---
def extract_pathogen_proteins(fasta_glob, protein_names):
    matches = {}
    for fasta in glob.glob(fasta_glob):
        for record in SeqIO.parse(fasta, "fasta"):
            header = record.description
            for pname in protein_names:
                if pname in header:
                    # Try to extract organism from header (after OS= or similar)
                    organism = None
                    if "OS=" in header:
                        organism = header.split("OS=")[1].split()[0]
                    else:
                        organism = pathogen_organism_map.get(pname, "Unknown")
                    matches[(pname, organism)] = str(record.seq)
    return matches

pathogen_matches = extract_pathogen_proteins(PATHOGEN_FASTA_GLOB, pathogen_proteins)

# --- 4. Sliding window peptide extraction ---
def sliding_window_peptides(seq, min_len=13, max_len=15):
    peptides = []
    for k in range(min_len, max_len+1):
        for i in range(len(seq) - k + 1):
            peptides.append(seq[i:i+k])
    return peptides

# --- 5. Write human peptide library ---
human_rows = []
for pname, seq in human_matches.items():
    peptides = sliding_window_peptides(seq)
    for pep in peptides:
        human_rows.append({"Human_peptides": pep, "Organism": "Homo sapiens", "Source_protein": pname})
pd.DataFrame(human_rows).to_csv(HUMAN_OUT, sep="\t", index=False)

# --- 6. Write pathogen peptide library ---
pathogen_rows = []
for (pname, organism), seq in pathogen_matches.items():
    # Exclude non-pathogen organisms (e.g., Homo, Mus)
    if organism.lower().startswith('homo') or organism.lower().startswith('mus'):
        continue
    peptides = sliding_window_peptides(seq)
    for pep in peptides:
        pathogen_rows.append({"Pathogen_peptides": pep, "Organism": organism, "Source_protein": pname})
pd.DataFrame(pathogen_rows).to_csv(PATHOGEN_OUT, sep="\t", index=False)

print(f"Done! {len(human_rows)} human peptides and {len(pathogen_rows)} pathogen peptides written.") 
