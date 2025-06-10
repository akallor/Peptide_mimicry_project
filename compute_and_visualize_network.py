#!/usr/bin/env python3
"""
Script to perform network clustering and visualization of peptide similarity networks.
Uses networkx for graph analysis and plotly for interactive visualization.
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import community  # python-louvain package
import argparse
import logging
from typing import Tuple, Dict
import colorsys

def load_network_data(network_file: Path, threshold: float = None) -> pd.DataFrame:
    """
    Load network data from TSV file.
    Optionally filter by similarity threshold.
    """
    df = pd.read_csv(network_file, sep='\t')
    if threshold is not None:
        df = df[df['weight'] >= threshold].copy()
        print(f"\nFiltering with threshold {threshold}:")
        print(f"Original pairs: {len(df)}")
        
        # Analyze similarity distribution
        print("\nSimilarity score distribution:")
        print(df['weight'].describe())
        
        # Count unique peptides
        unique_sources = df['source'].nunique()
        unique_targets = df['target'].nunique()
        print(f"\nUnique human peptides (hubs): {unique_sources}")
        print(f"Unique pathogen peptides (spokes): {unique_targets}")
        
        # Sample some high-similarity pairs
        print("\nSample of highest similarity pairs:")
        top_pairs = df.nlargest(5, 'weight')
        for _, row in top_pairs.iterrows():
            print(f"Human: {row['source']}")
            print(f"Pathogen: {row['target']}")
            print(f"Similarity: {row['weight']:.3f}")
            print("---")
    
    return df

def create_graph(df: pd.DataFrame) -> nx.Graph:
    """Create networkx graph from DataFrame."""
    G = nx.Graph()
    
    # Add nodes with type attributes
    sources = set(df['source'])
    targets = set(df['target'])
    
    # Add hub nodes (sources)
    for node in sources:
        G.add_node(node, node_type='hub')
    
    # Add spoke nodes (targets)
    for node in targets:
        G.add_node(node, node_type='spoke')
    
    # Add edges with weights
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], 
                  weight=row['weight'],
                  organism=row['organism'])
    
    return G

def get_node_positions(G: nx.Graph) -> Dict:
    """
    Compute node positions using ForceAtlas2 layout algorithm.
    Returns positions dict with scaled coordinates.
    """
    # Initial layout using spring layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Scale positions to reasonable range
    scaler = MinMaxScaler(feature_range=(-10, 10))
    pos_array = np.array(list(pos.values()))
    scaled_pos = scaler.fit_transform(pos_array)
    
    return {node: scaled_pos[i] for i, node in enumerate(pos.keys())}

def generate_distinct_colors(n: int) -> list:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.7 + np.random.random() * 0.3
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    return colors

def analyze_sequence_similarity(df: pd.DataFrame):
    """Analyze and print sequence similarity patterns."""
    # Calculate similarity statistics per hub
    hub_stats = df.groupby('source').agg({
        'target': 'count',
        'weight': ['mean', 'min', 'max']
    }).round(3)
    
    hub_stats.columns = ['num_matches', 'avg_similarity', 'min_similarity', 'max_similarity']
    hub_stats = hub_stats.sort_values('num_matches', ascending=False)
    
    print("\nTop 10 hubs by number of matches:")
    print(hub_stats.head(10))
    
    # Analyze similarity distribution
    similarity_ranges = pd.cut(df['weight'], 
                             bins=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                             labels=['0.70-0.75', '0.75-0.80', '0.80-0.85', 
                                   '0.85-0.90', '0.90-0.95', '0.95-1.00'])
    
    range_counts = similarity_ranges.value_counts().sort_index()
    print("\nDistribution of similarity scores:")
    print(range_counts)

def create_network_visualization(G: nx.Graph, output_file: Path):
    """Create interactive network visualization using plotly."""
    # Perform community detection
    communities = community.best_partition(G)
    num_communities = len(set(communities.values()))
    
    # Generate colors for communities
    community_colors = generate_distinct_colors(num_communities)
    
    # Get node positions
    pos = get_node_positions(G)
    
    # Prepare node traces for hubs and spokes
    hub_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'hub']
    spoke_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'spoke']
    
    # Create traces for nodes
    node_traces = []
    
    # Hub nodes
    x_hub = [pos[node][0] for node in hub_nodes]
    y_hub = [pos[node][1] for node in hub_nodes]
    hub_colors = [community_colors[communities[node]] for node in hub_nodes]
    
    node_traces.append(go.Scatter(
        x=x_hub, y=y_hub,
        mode='markers',
        marker=dict(
            size=15,
            color=hub_colors,
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        name='Hubs',
        text=[f"Hub: {node}<br>Community: {communities[node]}" for node in hub_nodes],
        hoverinfo='text'
    ))
    
    # Spoke nodes
    x_spoke = [pos[node][0] for node in spoke_nodes]
    y_spoke = [pos[node][1] for node in spoke_nodes]
    spoke_colors = [community_colors[communities[node]] for node in spoke_nodes]
    
    node_traces.append(go.Scatter(
        x=x_spoke, y=y_spoke,
        mode='markers',
        marker=dict(
            size=8,
            color=spoke_colors,
            symbol='diamond',
            line=dict(width=1, color='black')
        ),
        name='Spokes',
        text=[f"Spoke: {node}<br>Community: {communities[node]}" for node in spoke_nodes],
        hoverinfo='text'
    ))
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1.0))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces,
                   layout=go.Layout(
                       title='Peptide Similarity Network<br>Hubs and Spokes with Community Detection',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    # Save figure
    fig.write_html(output_file)
    
    # Print statistics
    print(f"\nNetwork Statistics:")
    print(f"Number of communities: {num_communities}")
    print(f"Number of hubs: {len(hub_nodes)}")
    print(f"Number of spokes: {len(spoke_nodes)}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Calculate and print community sizes
    community_sizes = pd.Series(communities.values()).value_counts()
    print("\nCommunity sizes:")
    print(community_sizes.sort_index())

def main():
    parser = argparse.ArgumentParser(description="Visualize peptide similarity network with clustering")
    parser.add_argument('--input', type=str, required=True,
                      help='Path to network TSV file')
    parser.add_argument('--output', type=str, default='network_visualization.html',
                      help='Output HTML file path (default: network_visualization.html)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Similarity threshold for filtering (e.g., 0.9 for highly similar sequences)')
    
    args = parser.parse_args()
    
    # Load and process data
    print("Loading network data...")
    df = load_network_data(Path(args.input), args.threshold)
    
    # Analyze similarity patterns
    analyze_sequence_similarity(df)
    
    print("\nCreating network graph...")
    G = create_graph(df)
    
    print("Generating visualization...")
    create_network_visualization(G, Path(args.output))
    
    print(f"\nVisualization saved to: {args.output}")

if __name__ == "__main__":
    main() 
