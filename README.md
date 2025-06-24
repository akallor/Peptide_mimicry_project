#Order to run the codes: 

1) Collect all bacterial and human data
collect_data.py

2) Generate peptide library (for pathogenic peptides)
peptide_library_generation.py

3) Compute the similarity using BLOSUM matrix
compute_peptide_similarity_new.py 
For TPUs, use:
compute_peptide_similarity_TPU_enhanced.py

If using ESM embeddings (not recommended due to time and computational constraints):
compute_peptide_similarity_ESM.py (CPU only) or compute_peptide_similarity_esm_GPU_enhanced.py (GPU only) or compute_peptide_similarity_ESM_TPU_enhanced.py (for TPU only)
                                                                                                                                                             
4) With the network output from the above steps, visualize interactively using:
compute_and_visualize_network.py

For 2D embeddings 1) & 2) remain the same, with the addition of binding affinity results (code to be added later). 2D embeddings = sequence identity + binding affinity elution rank
After 1) & 2) run:

Compute 2D network:
compute_2D_network_similarity.py (for BLOSUM) or compute_2D_network__similarity_ESM.py (not recommended due to unreasonable computational demands).

Visualization can be done with the same script ie compute_and_visualize_network.py

#Updated strategy: First run BLOSUM and binding affinity to get a 2D network (or just BLOSUM for 1D), filter by identity (threshold 0.8) and difference in elution rank (threshold difference <= 0.05)
then create a filtered network with the closest sequence identity and elution rank. Second, perform an ESM-based embedding on all the interacting partners in this filtered network to further filter out the interacting/mimicking
partners based on their sequence contexts.
