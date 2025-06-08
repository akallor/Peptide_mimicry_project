#!/usr/bin/env python3
"""
Module for graph-based clustering of similar peptides to identify mimicry networks.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
from community import community_louvain

class MimicryNetworkAnalyzer:
    def __init__(self, min_similarity: float = 0.8):
        self.data_dir = Path("data")
        self.similarity_dir = self.data_dir / "similarity_scores"
        self.output_dir = self.data_dir / "mimicry_networks"
        self.min_similarity = min_similarity
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_similarity_graph(self, similarity_data: pd.DataFrame) -> nx.Graph:
        """Build a graph from similarity scores."""
        G = nx.Graph()
        
        # Add nodes and edges
        for _, row in tqdm(similarity_data.iterrows(), desc="Building graph"):
            hla_seq = row['hla_sequence']
            pathogen_seq = row['pathogen_sequence']
            similarity = row['similarity_score']
            source = row['pathogen_source']
            
            # Add nodes with attributes
            if not G.has_node(hla_seq):
                G.add_node(hla_seq, type='hla')
            if not G.has_node(pathogen_seq):
                G.add_node(pathogen_seq, type='pathogen', source=source)
            
            # Add edge with similarity score
            G.add_edge(hla_seq, pathogen_seq, weight=similarity)
        
        return G

    def find_communities(self, G: nx.Graph) -> Dict:
        """Find communities in the graph using Louvain method."""
        # Apply Louvain community detection
        communities = community_louvain.best_partition(G)
        return communities

    def analyze_communities(self, G: nx.Graph, communities: Dict) -> List[Dict]:
        """Analyze the identified communities."""
        community_data = []
        
        # Group nodes by community
        community_groups = {}
        for node, community_id in communities.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(node)
        
        # Analyze each community
        for community_id, nodes in community_groups.items():
            if len(nodes) < 2:  # Skip single-node communities
                continue
            
            # Create subgraph for this community
            subgraph = G.subgraph(nodes)
            
            # Count node types
            hla_nodes = [n for n, attr in subgraph.nodes(data=True) if attr['type'] == 'hla']
            pathogen_nodes = [n for n, attr in subgraph.nodes(data=True) if attr['type'] == 'pathogen']
            
            # Get unique pathogen sources
            pathogen_sources = set(attr['source'] for _, attr in subgraph.nodes(data=True)
                                 if 'source' in attr)
            
            # Calculate average similarity within community
            similarities = [d['weight'] for _, _, d in subgraph.edges(data=True)]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            community_data.append({
                'community_id': community_id,
                'size': len(nodes),
                'n_hla': len(hla_nodes),
                'n_pathogen': len(pathogen_nodes),
                'avg_similarity': avg_similarity,
                'pathogen_sources': list(pathogen_sources),
                'nodes': nodes
            })
        
        return community_data

    def process_similarities(self):
        """Process all similarity files and generate mimicry networks."""
        # Load and combine all similarity files
        similarity_files = list(self.similarity_dir.glob("*_similarities.csv"))
        
        for sim_file in similarity_files:
            print(f"Processing {sim_file.name}...")
            
            # Load similarity data
            similarity_data = pd.read_csv(sim_file)
            
            # Build similarity graph
            G = self.build_similarity_graph(similarity_data)
            
            # Find communities
            communities = self.find_communities(G)
            
            # Analyze communities
            community_data = self.analyze_communities(G, communities)
            
            # Save results
            results_df = pd.DataFrame(community_data)
            output_file = self.output_dir / f"{sim_file.stem}_communities.csv"
            results_df.to_csv(output_file, index=False)
            
            # Save graph for visualization
            nx.write_gexf(G, self.output_dir / f"{sim_file.stem}_graph.gexf")
            
            print(f"Found {len(community_data)} communities")
            print(f"Average community size: {results_df['size'].mean():.2f}")
            print(f"Average similarity: {results_df['avg_similarity'].mean():.2f}")

def main():
    analyzer = MimicryNetworkAnalyzer()
    analyzer.process_similarities()

if __name__ == "__main__":
    main() 
