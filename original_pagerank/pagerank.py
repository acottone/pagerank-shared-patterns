import numpy as np
import time
from scipy.sparse import identity, csc_matrix
from scipy.sparse.linalg import spsolve
from visualizations import (create_visualizations)

def format_results(arr, G):
    """
    Format PageRank results.
    """
    in_degrees = np.array(G.sum(axis=0)).flatten() # Calculate sum of in-degrees (incoming links)
    out_degrees = np.array(G.sum(axis=1)).flatten() # Calculate sum of out-degrees (outgoing links)
    
    # List for storing results
    results = []
    for i, pr_value in enumerate(arr):
        results.append({
            'node_id': i + 1,
            'pagerank': pr_value,
            'in_degree': int(in_degrees[i]),
            'out_degree': int(out_degrees[i])
        })
    
    results.sort(key=lambda x: x['pagerank'], reverse=True) # Sort nodes by PageRank value in descending order
    return [f"Node {r['node_id']}: PR = {r['pagerank']:.4f}, In = {r['in_degree']}, Out = {r['out_degree']}"
            for r in results]

def print_pagerank_results(results):
    """
    Print PageRank results.
    """
    for result in results:
        print(result)

def pagerank1(G, p=0.85):
    """
    Compute the PageRank of a graph.
    INPUT :
    - G : Connectivity matrix of the graph .
    - p : Damping factor ( default : 0.85) .
    OUTPUT :
    - x : PageRank vector
    """
    n = G.shape[0] # Number of nodes
   
    # Calculate sum of out-degrees (outgoing links)
    c = G.sum(axis=0).A[0]
    
    # Generate diagonal matrix D with inverse out-degrees
    D = identity(n, format='csc')
    D.setdiag(1/np.where(c != 0, c, 1)) # Use 1 for nodes with value zero to avoid division by zero
    
    # Uniform vector for teleportation (random jumps)
    e = np.ones(n) / n
    
    # Identity matrix
    I = identity(n, format='csc')

    # Solve the linear system
    x = spsolve(I - p * G @ D, e)
    
    return x / np.sum(x) # Normalize results to sum to 1

def pagerank2(G, p=0.85, tol=1e-8, max_iter=100):
    """
    Compute the PageRank using inverse iteration.

    INPUT:
    - G: Sparse connectivity matrix
    - p: Damping factor
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations

    OUTPUT:
    - x: PageRank vector
    """
    n = G.shape[0] # Get number of nodes
    
    # Calculate sum of out-degrees (outgoing links)
    c = G.sum(axis=0).A[0]
    
    # Generate diagonal matrix D with inverse out-degrees
    D = identity(n, format='csc')
    D.setdiag(1/np.where(c != 0, c, 1)) # Use 1 for nodes with value zero to avoid division by zero
   
    # Calculate teleportation vector
    delta = (1-p)/n * np.ones(n)
    e = np.ones(n)
    
    # Initialize PageRank vector with uniform distribution
    x = np.ones(n) / n

    # Iterate until convergence or max iterations reached
    for _ in range(max_iter):
        x_old = x.copy()
        x = p * (G @ D @ x) + delta * np.sum(x) # Update PageRank vector
        x = x / np.sum(x) # Normalize to sum to 1
        
        # Check for convergence using L1 norm
        if np.linalg.norm(x - x_old, 1) < tol:
            break
            
    return x

def pagerank3(G, p=0.85, tol=1e-8, max_iter=100):
    """
    Compute the PageRank using power iteration.

    INPUT:
    - G: Sparse connectivity matrix
    - p: Damping factor
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations

    OUTPUT:
    - x: PageRank vector
    """
    n = G.shape[0] # Number of nodes
    
    # Calculate sum of out-degrees (outgoing links)
    c = G.sum(axis=0).A[0]
    
    # Generate diagonal matrix D with inverse out-degrees
    D = identity(n, format='csc')
    D.setdiag(1/np.where(c != 0, c, 1)) # Use 1 for nodes with value zero to avoid division by zero
    
    # Initialize uniform vector
    e = np.ones(n) / n
    x = e.copy() # Initialize PageRank vector with uniform distribution
    
    # Compute vector z for non-dangling nodes
    z = np.ones(n) / n
    z[np.where(c != 0)[0]] -= p / n # Adjust for nodes with outgoing links
    
    # Iterate until convergence or max iterations reached
    for _ in range(max_iter):
        x_old = x.copy()
        x = (p * (G @ D)) @ x + e * (z @ x) # Update PageRank vector
        
        # Check for convergence using L1 norm
        if np.linalg.norm(x - x_old, 1) < tol:
            break
    
    return x / np.sum(x) # Normalize to sum to 1

def load_graph(filename):
    """
    Load Stanford web data.

    INPUT:
    - filename: Path to the data file

    OUTPUT:
    - G: Sparse connectivity matrix
    """
    with open(filename, 'r') as f:
        header_lines = sum(1 for line in f if line.startswith('#'))
    
    # Load edge data, skipping header lines
    edges = np.loadtxt(filename, delimiter='\t', skiprows=header_lines, dtype=np.int64)
    n = max(edges.max(axis=0)) + 1 # Calculate number of nodes

    # Create sparse adjacency matrix in CSC format
    return csc_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))

def analyze_graph(G, sizes=[100, 1000, 10000, None]):
    """
    Analyze the graph using different PageRank methods.

    INPUT:
    - G: Sparse connectivity matrix
    - sizes: List of graph sizes to analyze

    OUTPUT:
    - results: Dictionary containing analysis results
    """
    # Store results
    results = {}
    computation_times = {'pr1': [], 'pr2': [], 'pr3': []}
    actual_sizes = []
    
    # Analyze different subsets of the graph
    for size in sizes:
        actual_size = G.shape[0] if size is None else min(size, G.shape[0])
        G_sub = G[:actual_size, :actual_size]
        actual_sizes.append(actual_size)
        
        size_label = "Full" if size is None else str(size)
        print(f"\nAnalyzing graph of size {size_label}...")
        print(f"Nodes: {G_sub.shape[0]}, Edges: {G_sub.nnz}")
        
        results[size_label] = {'stats': {'nodes': G_sub.shape[0], 'edges': G_sub.nnz}}
        pr_values_dict = {}
       
        # Compute PageRank using different methods
        for method_name, method_func in [
            ('pr1', pagerank1),
            ('pr2', pagerank2),
            ('pr3', pagerank3)
        ]:
            start_time = time.time()
            pr_values = method_func(G_sub)
            computation_times[method_name].append(time.time() - start_time)
            
            pr_values_dict[method_name] = pr_values
            results[size_label][method_name] = format_results(pr_values, G_sub)
            
            print(f"\nResults for {method_name}:")
            print_pagerank_results(results[size_label][method_name][:5])
        
        create_visualizations(G_sub, pr_values_dict, actual_size)
   
    return results

if __name__ == "__main__":
    labels = {
        'pr1': 'Direct Method',
        'pr2': 'Inverse Iteration Method',
        'pr3': 'Power Method'
    }
    
    try:
        #G_stanford = load_graph('web-Stanford.txt')
        #print(f"Loaded Stanford graph with {G_stanford.shape[0]} nodes and {G_stanford.nnz} edges")
        #analyze_graph(G_stanford)
        shared_patterns = load_graph('physician-shared-patient-patterns-2015-days180.txt')
        print(f"Loaded Stanford graph with {shared_patterns.shape[0]} nodes and {shared_patterns.nnz} edges")
        analyze_graph(shared_patterns)
    except FileNotFoundError:
        print("\nStanford dataset not found.")
