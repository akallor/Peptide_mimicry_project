#!/usr/bin/env python3
"""
Module for visualizing mimicry networks using an interactive dashboard.
"""

import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px

class NetworkVisualizer:
    def __init__(self):
        self.data_dir = Path("data")
        self.network_dir = self.data_dir / "mimicry_networks"
        self.app = Dash(__name__)
        self.setup_layout()

    def load_network_data(self):
        """Load all network data."""
        self.graphs = {}
        self.communities = {}
        
        # Load all graph and community files
        graph_files = list(self.network_dir.glob("*_graph.gexf"))
        
        for graph_file in graph_files:
            base_name = graph_file.stem.replace("_graph", "")
            
            # Load graph
            G = nx.read_gexf(graph_file)
            self.graphs[base_name] = G
            
            # Load corresponding communities
            community_file = self.network_dir / f"{base_name}_communities.csv"
            if community_file.exists():
                self.communities[base_name] = pd.read_csv(community_file)

    def create_network_figure(self, graph_name: str, selected_community: int = None) -> go.Figure:
        """Create network visualization for selected graph and community."""
        G = self.graphs[graph_name]
        
        if selected_community is not None:
            # Filter nodes for selected community
            community_data = self.communities[graph_name]
            community_nodes = eval(community_data[community_data['community_id'] == selected_community]['nodes'].iloc[0])
            G = G.subgraph(community_nodes)
        
        # Use force-directed layout
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Sequence: {node[0]}<br>Type: {node[1]['type']}")
            node_color.append('#1f77b4' if node[1]['type'] == 'hla' else '#ff7f0e')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_color,
                size=10,
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=f'Mimicry Network: {graph_name}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        
        return fig

    def setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("Molecular Mimicry Network Visualization"),
            
            html.Div([
                html.Label("Select Network:"),
                dcc.Dropdown(
                    id='network-selector',
                    options=[],
                    value=None
                ),
                
                html.Label("Select Community:"),
                dcc.Dropdown(
                    id='community-selector',
                    options=[],
                    value=None
                ),
                
                dcc.Graph(id='network-graph'),
                
                html.Div([
                    html.H3("Community Statistics"),
                    html.Div(id='community-stats')
                ])
            ])
        ])
        
        @self.app.callback(
            [Output('network-selector', 'options'),
             Output('network-selector', 'value')],
            Input('network-graph', 'figure')
        )
        def update_network_options(_):
            options = [{'label': name, 'value': name} for name in self.graphs.keys()]
            return options, options[0]['value'] if options else None
        
        @self.app.callback(
            [Output('community-selector', 'options'),
             Output('community-selector', 'value')],
            Input('network-selector', 'value')
        )
        def update_community_options(network):
            if network and network in self.communities:
                communities = self.communities[network]
                options = [{'label': f"Community {i}", 'value': i}
                          for i in communities['community_id'].unique()]
                return options, None
            return [], None
        
        @self.app.callback(
            Output('network-graph', 'figure'),
            [Input('network-selector', 'value'),
             Input('community-selector', 'value')]
        )
        def update_graph(network, community):
            if network:
                return self.create_network_figure(network, community)
            return go.Figure()
        
        @self.app.callback(
            Output('community-stats', 'children'),
            [Input('network-selector', 'value'),
             Input('community-selector', 'value')]
        )
        def update_stats(network, community):
            if network and community is not None:
                comm_data = self.communities[network]
                stats = comm_data[comm_data['community_id'] == community].iloc[0]
                return html.Div([
                    html.P(f"Size: {stats['size']}"),
                    html.P(f"HLA Peptides: {stats['n_hla']}"),
                    html.P(f"Pathogen Peptides: {stats['n_pathogen']}"),
                    html.P(f"Average Similarity: {stats['avg_similarity']:.3f}"),
                    html.P(f"Pathogen Sources: {', '.join(eval(stats['pathogen_sources']))}")
                ])
            return "Select a community to view statistics"

    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the visualization server."""
        self.load_network_data()
        self.app.run_server(debug=debug, port=port)

def main():
    visualizer = NetworkVisualizer()
    visualizer.run_server()

if __name__ == "__main__":
    main() 
