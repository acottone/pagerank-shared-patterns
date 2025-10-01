"""
Healthcare Network PageRank Analysis

This is an implementation of the Powers Iteration PageRank method for analyzing healthcare provider 
networks, with the goal of providing a comprehensive analysis of physician referral patterns with 
risk analysis.

Author: Angelina Cottone
Date: 09/05/2025
License: MIT

Key Features:
- Healthcare-optimized Power Iteration PageRank implementation
- Risk analysis with VaR/ES metrics
- Convergence stability analysis
- Interactive dashboard generation
- Memory-efficient sparse matrix operations
"""

import numpy as np
import time
import logging
from scipy.sparse import identity, csc_matrix
from visualizations2 import (create_healthcare_analysis_plots, create_risk_analysis_plots, 
                             create_convergence_analysis_plots, create_comprehensive_dashboard,
                             add_to_dashboard)

# Logging setup for analysis tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pagerank_analysis.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PageRankResult:
    """
    PageRank analysis results, stores all relevant metrics from PageRank computations including
    convergence information and performance metrics.

    Attributes:
        values (np.ndarray): PageRank scores for each node
        iterations (int): Number of iterations until convergence
        converged (bool): Whether algorithm converged within tolerance limits
        error (float): Final convergence error
        cpu_time (float): Computation time in seconds
        graph_size (int): Number of nodes in the graph
    """
    def __init__(self, values, iterations, converged, error, cpu_time, graph_size):
        self.values = values
        self.iterations = iterations
        self.converged = converged
        self.error = error
        self.cpu_time = cpu_time
        self.graph_size = graph_size

def validate_graph(G):
    """
    Validates graph matrix for PageRank analysis, ensures the graph is properly formatted as a sparse 
    matrix and meets requirements for PageRank computation.

    Args:
        G (scipy.sparse.csc_matrix): Input adjacency matrix

    Raises:
        TypeError: If G is not a csc_matrix
        ValueError: If G is not square or is empty
    """
    if not isinstance(G, csc_matrix):
        raise TypeError("Graph must be a scipy.sparse.csc_matrix")
    if G.shape[0] != G.shape[1]:
        raise ValueError("Graph matrix must be square")
    if G.shape[0] == 0:
        raise ValueError("Graph cannot be empty")

def analyze_network(G, index_to_npi=None, damping_factor=0.85, tolerance=1e-8):
    """
    Healthcare network analysis using Power Iteration PageRank, optimized for large-scale provider 
    datasets (1M+ nodes). 

    Args:
        G (scipy.sparse.csc_matrix): Adjacency matrix of provider network
        index_to_npi (dict, optional): Mapping from matrix indices to NPI numbers
        damping_factor (float): Damping parameter (default: 0.85)
        tolerance (float): Convergence tolerance (default: 1e-8)

    Returns:
        PageRankResult: Complete analysis results with metrics

    Raises:
        Exception: If computation fails
    """
    if index_to_npi is None:
        index_to_npi = {i: i for i in range(G.shape[0])}

    logger.info(f"Running PageRank analysis on {G.shape[0]:,} providers")

    try:
        pr_values, iterations, converged, convergence_error, cpu_time = pagerank(
            G, damping_factor=damping_factor, tolerance=tolerance
        )

        result = PageRankResult(
            values=pr_values,
            iterations=iterations,
            converged=converged,
            error=convergence_error,
            cpu_time=cpu_time,
            graph_size=G.shape[0]
        )

        logger.info(f"PageRank completed: {iterations} iterations, {cpu_time:.4f}s")
        return result

    except Exception as e:
        logger.error(f"PageRank failed: {e}")
        return {}

def pagerank(G, damping_factor=0.85, tolerance=1e-8, max_iterations=1000):
    """
    Power Iteration PageRank implementation with sparse matrix operations for memory efficiency, 
    includes optimizations such as damping factors and convergence criteria suitable for referral 
    networks.

    Args:
        G (scipy.sparse.csc_matrix): Adjacency matrix of provider referrals
        damping_factor (float): Probability of following link vs. random jump (default: 0.85)
        tolerance (float): L1 norm convergence threshold (default: 1e-8)
        max_iterations (int): Maximum iterations before stopping (default: 1000)

    Returns:
        tuple: (pagerank_values, iterations, converged, final_error, cpu_time)
            - pagerank_values (np.ndarray): Final PageRank scores
            - iterations (int): Number of iterations performed
            - converged (bool): Whether algorithm converged
            - final_error (float): Final L1 norm error
            - cpu_time (float): Computation time in seconds
    """
    validate_graph(G)
    start_time = time.time()
    n = G.shape[0] # Number of nodes (providers)

    # If the graph is empty, return empty results
    if n == 0:
        return np.array([]), 0, False, 1.0, 0.0

    c = G.sum(axis=0).A[0] # c = out-degrees for each node (outgoing referrals)

    D = identity(n, format='csc') # Sparse identity matrix
    D.setdiag(1/np.where(c != 0, c, 1)) # Avoid division by zero for isolated providers (no outgoing referrals)

    delta = (1-damping_factor)/n * np.ones(n) # delta = teleportation vector for random jumps

    x = np.ones(n) / n # Initialize with uniform distribution

    # Convergence tracking variables
    converged = False
    error = 0.0

    logger.info(f"Starting PageRank: {n:,} providers, damping={damping_factor}, tolerance={tolerance:.0e}")

    # Power iteration method: repeatedly apply PageRank equation until convergence
    for iteration in range(max_iterations):
        x_old = x.copy()  # Store previous iteration for convergence check

        # x = (probability of following link) + (probability of random jump)
        x = damping_factor * (G @ D @ x) + delta * np.sum(x)

        # Sum to 1 for probability distribution
        x = x / np.sum(x)

        # L1 norm error between iterations
        error = np.linalg.norm(x - x_old, 1)

        # Check for convergence
        if error < tolerance:
            converged = True
            break

        if iteration % 100 == 0 and iteration > 0:
            logger.debug(f"Iteration {iteration}: error = {error:.2e}")

    cpu_time = time.time() - start_time

    if not converged:
        logger.warning(f"PageRank did not converge after {max_iterations} iterations. Final error: {error:.2e}")
    else:
        logger.info(f"PageRank converged in {iteration + 1} iterations, time={cpu_time:.4f}s, error={error:.2e}")

    return x, iteration + 1, converged, error, cpu_time

def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for inequality analysis.

    Args:
        values (np.ndarray): Array of values to analyze

    Returns:
        float: Gini coefficient
    """
    if len(values) == 0: # Avoid division by zero
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values) # Sum of sorted values

    # Gini coefficient formula
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

def analyze_risk(G, risk_scores, index_to_npi=None, damping_factor=0.95, tolerance=1e-8, max_iterations=1000):
    """
    Risk propagation analysis to see how individual provider risk scores flow through the network.

    Args:
        G (scipy.sparse.csc_matrix): Adjacency matrix of provider referrals
        risk_scores (np.ndarray): Initial risk scores for each provider
        index_to_npi (dict, optional): Mapping from matrix indices to NPI numbers
        damping_factor (float): Damping parameter (default: 0.95)
        tolerance (float): Convergence tolerance (default: 1e-8)
        max_iterations (int): Maximum iterations before stopping (default: 1000)
        
    Returns:
        dict: Risk analysis results containing:
            - propagated_risk_scores: Array of risk scores for each provider
            - original_risk_scores: Initial risk scores provided
            - amplification_factors: How much each provider's risk was amplified
            - high_risk_providers: List of providers exceeding 90th percentile risk
            - extreme_risk_providers: List of providers exceeding 99th percentile risk
    """
    validate_graph(G)
    if len(risk_scores) != G.shape[0]:
        raise ValueError("Length of risk scores must match graph size")

    logger.info("Starting risk propagation analysis")

    # Sum to 1 for probability distribution
    risk_scores = risk_scores / np.sum(risk_scores)

    n = G.shape[0]
    c = G.sum(axis=0).A[0]

    D = identity(n, format='csc')
    D.setdiag(1 / np.where(c != 0, c, 1))

    x = risk_scores.copy() # Current risk distribution (starts with initial scores)

    delta = (1 - damping_factor) * risk_scores

    for iteration in range(max_iterations):
        x_old = x.copy()  # Store previous risk distribution

        # Risk flows through network, with some remaining at origin
        x = damping_factor * (G @ D @ x) + delta

        # Check convergence
        if np.linalg.norm(x - x_old, 1) < tolerance:
            break

    # Risk thresholds for provider categorization
    risk_threshold_90 = np.percentile(x, 90) # Top 10% risk threshold
    risk_threshold_95 = np.percentile(x, 95) # Top 5% risk threshold
    risk_threshold_99 = np.percentile(x, 99) # Top 1% risk threshold (extreme)

    # Identify high-risk provider indices for detailed analysis
    high_risk_indices = np.where(x > risk_threshold_90)[0] # Top 10% providers
    extreme_risk_indices = np.where(x > risk_threshold_99)[0] # Top 1% providers

    # Risk amplification, how much provide risk was amplified by network effects
    amplification_factors = np.where(risk_scores > 0, x / risk_scores, 1.0) # amp > 1 network increased risk, < 1 network reduced risk

    # Value at Risk, max expected loss at given confidence level
    var_95 = np.percentile(x, 95) # 95% of providers have risk below this level
    var_99 = np.percentile(x, 99)  # 99% of providers have risk below this level

    # Expected shortfall, avg risk of providers exceeding VaR threshold
    es_95 = np.mean(x[x >= var_95]) if np.any(x >= var_95) else var_95 # Avg risk of worst 5%
    es_99 = np.mean(x[x >= var_99]) if np.any(x >= var_99) else var_99 # Avg risk of worst 1%

    results = {
        'propagated_risk_scores': x,
        'original_risk_scores': risk_scores,
        'amplification_factors': amplification_factors,
        'high_risk_providers': [],
        'extreme_risk_providers': [],
        'risk_statistics': {
            'mean_risk': float(np.mean(x)),
            'median_risk': float(np.median(x)),
            'std_risk': float(np.std(x)),
            'max_risk': float(np.max(x)),
            'min_risk': float(np.min(x)),

            'risk_concentration_90': float(np.sum(x[high_risk_indices]) / np.sum(x)),
            'risk_concentration_95': float(np.sum(x[x > risk_threshold_95]) / np.sum(x)),
            'risk_concentration_99': float(np.sum(x[x > risk_threshold_99]) / np.sum(x)),

            'var_95': float(var_95),
            'var_99': float(var_99),
            'expected_shortfall_95': float(es_95),
            'expected_shortfall_99': float(es_99),

            'mean_amplification': float(np.mean(amplification_factors)),
            'max_amplification': float(np.max(amplification_factors)),
            'amplification_gini': float(calculate_gini_coefficient(amplification_factors)),

            'high_risk_count': len(high_risk_indices),
            'extreme_risk_count': len(extreme_risk_indices),
            'high_risk_percentage': float(len(high_risk_indices) / len(x) * 100),
            'extreme_risk_percentage': float(len(extreme_risk_indices) / len(x) * 100),

            'risk_threshold_90': float(risk_threshold_90),
            'risk_threshold_95': float(risk_threshold_95),
            'risk_threshold_99': float(risk_threshold_99)
        }
    }

    if index_to_npi:
        for idx in high_risk_indices:
            results['high_risk_providers'].append({
                'npi': index_to_npi.get(idx, idx),
                'risk_score': float(x[idx]),
                'original_risk': float(risk_scores[idx]),
                'amplification_factor': float(amplification_factors[idx]),
                'risk_percentile': float(np.sum(x <= x[idx]) / len(x) * 100),
                'provider_index': int(idx)
            })

        for idx in extreme_risk_indices:
            results['extreme_risk_providers'].append({
                'npi': index_to_npi.get(idx, idx),
                'risk_score': float(x[idx]),
                'original_risk': float(risk_scores[idx]),
                'amplification_factor': float(amplification_factors[idx]),
                'risk_percentile': float(np.sum(x <= x[idx]) / len(x) * 100),
                'provider_index': int(idx)
            })

        results['high_risk_providers'].sort(key=lambda p: p['risk_score'], reverse=True)
        results['extreme_risk_providers'].sort(key=lambda p: p['risk_score'], reverse=True)

    logger.info(f"Risk propagation complete. {len(high_risk_indices)} providers identified")
    return results

def analyze_risk_propagation(G, risk_scores, index_to_npi=None, damping_factor=0.95, tolerance=1e-8, max_iterations=1000):
    return analyze_risk(G, risk_scores, index_to_npi, damping_factor, tolerance, max_iterations)

def analyze_provider_network(G, specialties=None, index_to_npi=None, damping_factor=0.85, tolerance=1e-8):
    """
    Comprehensive analysis of healthcare provider network structure and key players.

    Identifies different types of influential providers based on network topology:
    - Hub providers: High outgoing referrals (refer many patients out)
    - Authority providers: High incoming referrals (receive many referrals)
    - Central providers: High PageRank scores (overall network importance)

    Args:
        G (scipy.sparse.csc_matrix): Provider referral network adjacency matrix
        specialties (dict, optional): Provider specialty mappings (currently unused)
        index_to_npi (dict, optional): Mapping from matrix indices to NPI numbers
        damping_factor (float): PageRank damping parameter (default: 0.85)
        tolerance (float): PageRank convergence tolerance (default: 1e-8)

    Returns:
        dict: Comprehensive network analysis results including provider categories and metrics
    """
    validate_graph(G)
    logger.info("Starting provider network analysis")

    if index_to_npi is None:
        index_to_npi = {i: i for i in range(G.shape[0])}

    # PageRank centrality scores for all providers, identifies overall network importance
    centrality_scores, iterations, converged, error, cpu_time = pagerank(
        G, damping_factor=damping_factor, tolerance=tolerance
    )

    logger.info(f"PageRank analysis: {iterations} iterations, converged={converged}, time={cpu_time:.4f}s")

    # Degree centralities for provider role identification
    # in_degree = number of incoming referrals
    # out_degree = number of outgoing referrals
    in_degrees = np.array(G.sum(axis=0)).flatten() # Authority measure: receives referrals
    out_degrees = np.array(G.sum(axis=1)).flatten() # Hub measure: sends referrals

    # Thresholds for identifying key providers (top 10% in each category), identify the most influential providers
    hub_threshold = np.percentile(out_degrees, 90) # Top 10% by outgoing referrals
    authority_threshold = np.percentile(in_degrees, 90) # Top 10% by incoming referrals
    central_threshold = np.percentile(centrality_scores, 90) # Top 10% by PageRank centrality

    # Identify key provider categories based on network roles
    hub_providers = np.where(out_degrees > hub_threshold)[0] # PCPs, specialists who refer frequently
    authority_providers = np.where(in_degrees > authority_threshold)[0] # Specialists, hospitals who receive many referrals
    central_providers = np.where(centrality_scores > central_threshold)[0] # Overall network influencers

    results = {
        'centrality_scores': centrality_scores,
        'network_metrics': {
            'total_providers': G.shape[0],
            'total_referrals': G.nnz,
            'network_density': float(G.nnz / (G.shape[0] * (G.shape[0] - 1))) if G.shape[0] > 1 else 0.0,

            'average_referrals_out': float(np.mean(out_degrees)),
            'median_referrals_out': float(np.median(out_degrees)),
            'std_referrals_out': float(np.std(out_degrees)),
            'max_referrals_out': int(np.max(out_degrees)),
            'referral_concentration_out': float(np.sum(out_degrees > np.percentile(out_degrees, 90)) / G.shape[0]),

            'average_referrals_in': float(np.mean(in_degrees)),
            'median_referrals_in': float(np.median(in_degrees)),
            'std_referrals_in': float(np.std(in_degrees)),
            'max_referrals_in': int(np.max(in_degrees)),
            'referral_concentration_in': float(np.sum(in_degrees > np.percentile(in_degrees, 90)) / G.shape[0]),

            'isolated_providers': int(np.sum((in_degrees + out_degrees) == 0)),
            'hub_providers': int(np.sum((in_degrees + out_degrees) > np.percentile(in_degrees + out_degrees, 95))),
            'connectivity_gini': float(calculate_gini_coefficient(in_degrees + out_degrees)),
            'referral_imbalance_ratio': float(np.mean(np.where(in_degrees > 0, out_degrees / (in_degrees + 1e-10), out_degrees))),

            'pagerank_iterations': iterations,
            'pagerank_converged': converged,
            'pagerank_computation_time': cpu_time,
            'pagerank_damping_factor': damping_factor,
            'pagerank_tolerance': tolerance
        },
        'key_providers': {
            'hubs': [],
            'authorities': [],
            'central': []
        }
    }

    for provider_type, indices in [
        ('hubs', hub_providers),
        ('authorities', authority_providers),
        ('central', central_providers)
    ]:
        for idx in indices:
            provider_info = {
                'index': int(idx),
                'centrality_score': float(centrality_scores[idx]),
                'referrals_out': int(out_degrees[idx]),
                'referrals_in': int(in_degrees[idx])
            }

            if index_to_npi:
                provider_info['npi'] = index_to_npi.get(idx, idx)
            if specialties:
                provider_info['specialty'] = specialties.get(idx, 'Unknown')

            results['key_providers'][provider_type].append(provider_info)

    logger.info(f"Provider network analysis complete. Identified {len(hub_providers)} hubs, {len(authority_providers)} authorities, {len(central_providers)} central providers")
    return results

def analyze_convergence(G, tolerance_range=None):
    """
    Analyze PageRank algorithm performance across different tolerance levels.

    Args:
        G (scipy.sparse.csc_matrix): Adjacency matrix of provider referrals
        tolerance_range (list, optional): List of tolerance levels to test (default: [1e-4, 1e-6, 1e-8, 1e-10, 1e-12])

    Returns:
        dict: Convergence analysis results for each tolerance level
    """
    if tolerance_range is None:
        tolerance_range = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

    # Store convergence results for each tolerance level
    convergence_data = {}

    # Test convergence behavior across different tolerance levels
    for tol in tolerance_range:
        try:
            start_time = time.time() # Start timer

            n = G.shape[0]
            c = G.sum(axis=0).A[0]

            D = identity(n, format='csc')
            D.setdiag(1/np.where(c != 0, c, 1))

            delta = (1-0.85)/n * np.ones(n)
            x = np.ones(n) / n

            converged = False
            error = 0.0
            max_iter = 1000  # Max iterations to prevent infinite loops

            # Power iteration method with current tolerance level
            for iteration in range(max_iter):
                x_old = x.copy()

                x = 0.85 * (G @ D @ x) + delta * np.sum(x)
                x = x / np.sum(x)

                error = np.linalg.norm(x - x_old, 1)

                if error < tol:
                    converged = True
                    break

            cpu_time = time.time() - start_time

            # Store comprehensive results for analysis and visualization
            convergence_data[tol] = {
                'iterations': iteration + 1, # Number of iterations required
                'cpu_time': cpu_time, # Computation time in seconds
                'converged': converged, # Whether algorithm converged
                'final_error': float(error) # Final L1 norm error achieved
            }

            logger.info(f"Tolerance {tol}: {iteration + 1} iterations, converged={converged}, error={error:.2e}")

        except Exception as e:
            logger.warning(f"Failed at tolerance {tol}: {e}")
            convergence_data[tol] = None

    return convergence_data



def load_graph(filename):
    """
    Load provider referral network from file and construct adjacency matrix.

    Args:
        filename (str): Path to file containing referral network data

    Returns:
        tuple: (G, index_to_npi)
            - G (scipy.sparse.csc_matrix): Adjacency matrix of provider referrals
            - index_to_npi (dict): Mapping from matrix indices to NPI numbers
    """
    logger.info(f"Loading file: {filename}")
    try:
        edges_raw = np.loadtxt(filename, delimiter=',', dtype=np.int64)
        logger.info("File loaded successfully")

        edges = edges_raw[:, :2]
        unique_npis = np.unique(edges)
        logger.info(f"Found {len(unique_npis)} unique NPIs")

        npi_to_index = {int(npi): idx for idx, npi in enumerate(unique_npis)}
        index_to_npi = {idx: int(npi) for npi, idx in npi_to_index.items()}

        logger.info("Mapping NPIs to indices")

        edges_indexed = np.array([[npi_to_index[int(src)], npi_to_index[int(dst)]]
                                 for src, dst in edges])
        logger.info("Edges relabeled")

        n = len(unique_npis)
        G = csc_matrix((np.ones(len(edges_indexed)),
                       (edges_indexed[:, 0], edges_indexed[:, 1])),
                      shape=(n, n))

        logger.info("Graph construction complete")
        return G, index_to_npi

    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        raise



def run_healthcare_analysis(G, index_to_npi):
    """
    Run focused healthcare network analysis on the full dataset.

    Args:
        G (scipy.sparse.csc_matrix): Network adjacency matrix
        index_to_npi (dict): Mapping from matrix indices to NPI numbers

    Returns:
        dict: Analysis results
    """
    logger.info(f"Starting healthcare analysis on full dataset: {G.shape[0]:,} providers")

    # Step 1: Core PageRank Analysis
    pagerank_result = analyze_network(G, index_to_npi)

    # Step 2: Provider Network Structure Analysis
    provider_result = analyze_provider_network(G, index_to_npi=index_to_npi)

    # Step 3: Risk Score Construction
    in_degrees = np.array(G.sum(axis=0)).flatten()
    out_degrees = np.array(G.sum(axis=1)).flatten()
    total_degrees = in_degrees + out_degrees

    # Connectivity risk, higher connectivity = higher exposure to network-wide risk
    connectivity_risk = total_degrees / np.max(total_degrees) if np.max(total_degrees) > 0 else np.zeros(G.shape[0])

    # Measures imbalance in referral patterns
    referral_imbalance = np.where(in_degrees > 0, out_degrees / (in_degrees + 1e-10), out_degrees)
    referral_imbalance = referral_imbalance / np.max(referral_imbalance) if np.max(referral_imbalance) > 0 else np.zeros(G.shape[0])

    # Providers with fewer connections
    isolation_risk = 1.0 / (1.0 + total_degrees)
    isolation_risk = isolation_risk / np.max(isolation_risk) if np.max(isolation_risk) > 0 else np.zeros(G.shape[0])

    # Use equal weights for all risk components - simple and unbiased approach
    base_risk = 0.001 # Minimum risk level for all providers
    risk_scores = (base_risk +
                  (1/3) * connectivity_risk +
                  (1/3) * referral_imbalance +
                  (1/3) * isolation_risk)

    # Normalize to probability distribution for mathematical consistency
    risk_scores = risk_scores / np.sum(risk_scores)

    # Step 4: Risk Propagation Analysis
    risk_result = analyze_risk_propagation(G, risk_scores, index_to_npi)

    # Step 5: Algorithm Performance Analysis
    tolerance_levels = [1e-4, 1e-6, 1e-8, 1e-10]
    convergence_result = analyze_convergence(G, tolerance_levels)

    return {
        'pagerank': pagerank_result,
        'provider_network': provider_result,
        'risk_analysis': risk_result,
        'convergence': convergence_result
    }

if __name__ == "__main__":
    """
    Main execution block for healthcare network PageRank analysis.

    1. Load healthcare network data from CMS dataset
    2. Run comprehensive PageRank and risk analysis
    3. Generate static and interactive visualizations
    4. Display key findings and performance metrics
    """
    start_time = time.time()

    try:
        print("Healthcare Network PageRank Analysis")
        print("=" * 50)

        print("Loading network data")
        graph_data = load_graph('physician-shared-patient-patterns-2015-days180.txt')

        if graph_data is None:
            print("Failed to load graph data")
            exit(1)

        G, index_to_npi = graph_data
        print(f"Loaded network: {G.shape[0]:,} providers, {G.nnz:,} referral relationships")

        print("\nRunning network analysis...")
        results = run_healthcare_analysis(G, index_to_npi)

        pagerank_result = results['pagerank']
        provider_results = results['provider_network']
        risk_results = results['risk_analysis']
        convergence_results = results['convergence']

        print(f"\nNetwork Analysis Results:")
        metrics = provider_results['network_metrics']
        print(f"  • Total Providers: {metrics['total_providers']:,}")
        print(f"  • Total Referrals: {metrics['total_referrals']:,}")
        print(f"  • Average Referrals Out: {metrics['average_referrals_out']:.2f}")
        print(f"  • Average Referrals In: {metrics['average_referrals_in']:.2f}")
        print(f"  • Network Density: {metrics['network_density']:.2e}")

        print(f"\nKey Provider Categories:")
        for provider_type, providers in provider_results['key_providers'].items():
            print(f"  • {provider_type.title()}: {len(providers)} providers")
            if providers:
                top_provider = providers[0]
                print(f"    └─ Top Provider: NPI {top_provider['npi']}")
                print(f"       Centrality: {top_provider['centrality_score']:.4f}")
                print(f"       Referrals Out/In: {top_provider['referrals_out']}/{top_provider['referrals_in']}")

        print(f"\nPageRank Analysis:")
        print(f"  • Converged: {'Yes' if pagerank_result.converged else 'No'}")
        print(f"  • Iterations: {pagerank_result.iterations}")
        print(f"  • Computation Time: {pagerank_result.cpu_time:.4f}s")
        print(f"  • Final Error: {pagerank_result.error:.2e}")

        print(f"\nRisk Analysis Results:")
        stats = risk_results['risk_statistics']
        print(f"  • Mean Risk Score: {stats['mean_risk']:.6f}")
        print(f"  • Maximum Risk Score: {stats['max_risk']:.6f}")
        print(f"  • Risk Concentration (top 10%): {stats['risk_concentration_90']:.2%}")
        print(f"  • High-risk Providers: {len(risk_results['high_risk_providers'])}")
        print(f"  • VaR 95%: {stats['var_95']:.6f}")
        print(f"  • Expected Shortfall 95%: {stats['expected_shortfall_95']:.6f}")

        if risk_results['high_risk_providers']:
            print(f"\nTop 3 High-Risk Providers:")
            for i, provider in enumerate(risk_results['high_risk_providers'][:3]):
                print(f"    {i+1}. NPI {provider['npi']}")
                print(f"       Risk Score: {provider['risk_score']:.6f}")
                print(f"       Amplification: {provider['amplification_factor']:.2f}x")

        print(f"\nConvergence Analysis:")
        print(f"{'Tolerance':<12} {'Iterations':<12} {'Time (s)':<10} {'Converged':<10}")
        print("-" * 55)

        for tol, data in convergence_results.items():
            if data:
                status = "Yes" if data['converged'] else "No"
                print(f"{tol:<12.0e} {data['iterations']:<12} {data['cpu_time']:<10.4f} {status:<10}")
            else:
                print(f"{tol:<12.0e} {'Failed':<12} {'-':<10} {'No':<10}")

        successful_results = [data for data in convergence_results.values() if data and data['converged']]
        if len(successful_results) > 1:
            iterations = [data['iterations'] for data in successful_results]
            times = [data['cpu_time'] for data in successful_results]
            print(f"\n Performance Summary:")
            print(f"    • Iteration Range: {min(iterations)} - {max(iterations)}")
            print(f"    • Time Range: {min(times):.4f}s - {max(times):.4f}s")
            print(f"    • Average Iterations: {np.mean(iterations):.1f}")

        # Create visualizations
        try:
            print(f"\nCreating visualizations...")

            create_healthcare_analysis_plots(G, pagerank_result, provider_results)
            create_risk_analysis_plots(risk_results, "Healthcare Risk Analysis")
            create_convergence_analysis_plots(convergence_results)

            add_to_dashboard('pagerank', {'Healthcare PageRank': pagerank_result.values}, size='Full')
            add_to_dashboard('risk', risk_results)
            add_to_dashboard('convergence', convergence_results)
            create_comprehensive_dashboard()

            print(f"Visualizations saved to ./visualizations/ folder")
            print(f"Interactive dashboard: ./visualizations/healthcare_network_dashboard.html")
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")

        print(f"\n" + "="*60)
        print(f"HEALTHCARE NETWORK ANALYSIS COMPLETE")
        print(f"="*60)
        print(f"Dataset: Physician Shared Patient Patterns (2015)")
        print(f"Network Size: {G.shape[0]:,} providers, {G.nnz:,} referrals")
        print(f"PageRank: {pagerank_result.iterations} iterations, {pagerank_result.cpu_time:.3f}s")
        print(f"High-Risk Providers: {len(risk_results['high_risk_providers'])}")
        print(f"Risk Concentration: {risk_results['risk_statistics']['risk_concentration_90']:.1%}")
        print(f"="*60)

    except FileNotFoundError:
        print("Dataset not found")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

    print(f"\n[Checkpoint] Total runtime: {time.time() - start_time:.2f} seconds")