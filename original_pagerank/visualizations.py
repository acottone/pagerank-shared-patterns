import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix

def create_visualizations(G, pr_values_dict, size):
    """
    Create visualizations for PageRank analysis.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
    
    method_labels = {
        'pr1': 'Direct Method',
        'pr2': 'Inverse Iteration',
        'pr3': 'Power Method'
    }
    
    colors = {
        'pr1': 'blue',
        'pr2': 'green',
        'pr3': 'red'
    }
    
    # Only create network visualization for smaller subsets
    if size <= 1000:
        ax1 = fig.add_subplot(gs[0])
        nx_graph = nx.from_scipy_sparse_array(G, create_using=nx.DiGraph)
        pos = nx.spring_layout(nx_graph) # Calculate node positions
        
        # Draw nodes for each PageRank method
        for method, values in pr_values_dict.items():
            node_sizes = [v * 5000 for v in values] # Adjust node sizes
            nx.draw_networkx_nodes(nx_graph, pos, 
                                 node_size=node_sizes,
                                 node_color=colors[method],
                                 alpha=0.3,
                                 label=method_labels[method])
        
        # Draw edges with arrows
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, arrows=True)
        ax1.set_title(f'Network Graph Comparison (n={size})')
        ax1.axis('off')
        ax1.legend()
    
    ax2 = fig.add_subplot(gs[1] if size <= 1000 else gs[0])
    
    # Plot PageRank values for each method
    for method, values in pr_values_dict.items():
        sorted_indices = np.argsort(values)[::-1]
        ax2.plot(np.arange(len(values)), 
                values[sorted_indices],
                color=colors[method],
                label=method_labels[method],
                linewidth=2)
    
    ax2.set_xlabel('Node Rank')
    ax2.set_ylabel('PageRank Value')
    ax2.set_title(f'PageRank Distribution Comparison (n={size})')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  
    ax2.legend()
    plt.tight_layout()
    
    if size == G.shape[0]:
        size_label = f"_{G.shape[0]}"
    else:
        size_label = str(size)
    
    plt.savefig(f'pagerank_analysis_comparison_{size_label}.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()