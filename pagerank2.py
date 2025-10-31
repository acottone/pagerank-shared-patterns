"""
Healthcare Network PageRank Analysis

This project applies the PageRank algorithm to healthcare provider networks to identify influential providers
and assess how risk spreads through the network.

This builds on the pagerank3 implementation (power iteration) from my original MAT 167 project and applies it to
healthcare data.

Author: Angelina Cottone
Date: August 2025
License: MIT
"""

import numpy as np
import time
import logging
from scipy.sparse import identity, csc_matrix
from visualizations2 import (create_healthcare_analysis_plots, create_risk_analysis_plots, 
                             create_convergence_analysis_plots, create_comprehensive_dashboard,
                             add_to_dashboard)

# Set up logging to track analysis progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pagerank_analysis.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PageRankResult:
    """
    Keeps track of all the Pagerank results in one place.

    Attributes:
        values (np.ndarray): The actual PageRank scores for each provider
        iterations (int): How many times we had to iterate before it converged
        converged (bool): Did it converge?
        error (float): How close we got to the "true" answer
        cpu_time (float): How long it took to run (useful for big networks)
        graph_size (int): Number of providers we analyzed
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
    Quick sanity check to make sure the graph isn't totally broken.

    Args:
        G (scipy.sparse.csc_matrix): The network as a sparse matrix

    Raises:
        TypeError: If it's not the right type of matrix
        ValueError: If the matrix is invalid (not square, empty, etc.)
    """
    if not isinstance(G, csc_matrix):
        raise TypeError("Graph needs to be a scipy.sparse.csc_matrix")
    if G.shape[0] != G.shape[1]:
        raise ValueError("Graph must be square")
    if G.shape[0] == 0:
        raise ValueError("Graph is empty")

def analyze_network(G, index_to_npi=None, damping_factor=0.85, tolerance=1e-8):
    """
    This is the main function that runs PageRank on a healthcare network.

    Args:
        G (scipy.sparse.csc_matrix): The provider network as a sparse matrix
        index_to_npi (dict): Dictionary that maps matrix positions to actual provider IDs
        damping_factor (float): How much we trust the network structure vs random jumps (default: 0.85)
        tolerance (float): How precise we want the answer (smaller = more precise but slower, default: 1e-8)

    Returns:
        PageRankResult: All the results bundled up in one object

    Raises:
        Exception: If something goes wrong during computation
    """
    # If no provider ID mapping is given, just use numbers
    if index_to_npi is None:
        index_to_npi = {i: i for i in range(G.shape[0])}

    logger.info(f"Starting PageRank on {G.shape[0]:,} providers...")

    try:
        # Run pagerank algorithm
        pr_values, iterations, converged, convergence_error, cpu_time = pagerank(
            G, damping_factor=damping_factor, tolerance=tolerance
        )

        # Store results
        result = PageRankResult(
            values=pr_values,
            iterations=iterations,
            converged=converged,
            error=convergence_error,
            cpu_time=cpu_time,
            graph_size=G.shape[0]
        )

        logger.info(f"Done. Took {iterations} iterations and {cpu_time:.4f} seconds")
        return result

    except Exception as e:
        logger.error(f"Oops, PageRank crashed: {e}")
        return {}

def pagerank(G, damping_factor=0.85, tolerance=1e-8, max_iterations=1000):
    """
    The actual PageRank algorithm - this is the enhanced version of pagerank3 from my MAT 167 project.

    Args:
        G (scipy.sparse.csc_matrix): The network as a sparse matrix (rows = providers, columns = who they refer to)
        damping_factor (float): How much we trust the network vs random surfing (default: 0.85)
        tolerance (float): When to stop iterating (smaller = more accurate but slower, default: 1e-8)
        max_iterations (int): Give up after this many tries if it doesn't converge (default: 1000)

    Returns:
        tuple: A tuple with all the results:
            - pagerank_values (np.ndarray): The actual PageRank scores for each provider
            - iterations (int): How many times we had to iterate
            - converged (bool): Did it actually converge or did we give up?
            - final_error (float): How close we got to the "true" answer
            - cpu_time (float): How long it took to run
    """
    validate_graph(G)
    start_time = time.time()
    n = G.shape[0]  # How many providers we're analyzing

    # If network empty
    if n == 0:
        return np.array([]), 0, False, 1.0, 0.0

    # Count how many referrals each provider makes (out-degree)
    c = G.sum(axis=0).A[0]

    # Create diagonal matrix for normalization (handles providers with no outgoing referrals)
    D = identity(n, format='csc')
    D.setdiag(1/np.where(c != 0, c, 1))  # Avoid dividing by zero

    # Random jump vector, where you go if you randomly teleport
    delta = (1-damping_factor)/n * np.ones(n)

    # Start with everyone having equal importance
    x = np.ones(n) / n

    # Keep track of whether we've converged
    converged = False
    error = 0.0

    logger.info(f"Starting PageRank: {n:,} providers, damping={damping_factor}, tolerance={tolerance:.0e}")

    # Keep iterating until we converge
    for iteration in range(max_iterations):
        x_old = x.copy()  # Remember old values

        # Follow network links + random jumps
        x = damping_factor * (G @ D @ x) + delta * np.sum(x)

        # Make sure it's still a proper probability distribution
        x = x / np.sum(x)

        # Calculate convergence error
        error = np.linalg.norm(x - x_old, 1)

        # Check convergence
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
    Calculates Gini coefficient to measure inequality in value distribution.

    Args:
        values (np.ndarray): Array of values to analyze

    Returns:
        float: Gini coefficient (0 = perfect equality, 1 = maximum inequality)
    """
    if len(values) == 0:  # Handle empty array
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)

    # Standard Gini coefficient formula
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

def analyze_risk(G, risk_scores, index_to_npi=None, damping_factor=0.95, tolerance=1e-8, max_iterations=1000):
    """
    Analyzes how risk propagates through the provider network.

    Args:
        G (scipy.sparse.csc_matrix): Provider referral network adjacency matrix
        risk_scores (np.ndarray): Initial risk scores for each provider
        index_to_npi (dict): Maps matrix indices to NPI numbers
        damping_factor (float): Risk retention factor (default: 0.95)
        tolerance (float): Convergence tolerance (default: 1e-8)
        max_iterations (int): Maximum iterations (default: 1000)

    Returns:
        dict: Risk analysis results with propagated scores, statistics, and high-risk providers
    """
    validate_graph(G)
    if len(risk_scores) != G.shape[0]:
        raise ValueError("Length of risk scores must match graph size")

    logger.info("Starting risk propagation analysis")

    # Normalize to probability distribution
    risk_scores = risk_scores / np.sum(risk_scores)

    n = G.shape[0]
    c = G.sum(axis=0).A[0]

    D = identity(n, format='csc')
    D.setdiag(1 / np.where(c != 0, c, 1))

    x = risk_scores.copy()  # Current risk distribution

    delta = (1 - damping_factor) * risk_scores

    for iteration in range(max_iterations):
        x_old = x.copy()

        # Risk flows through network with some staying at origin
        x = damping_factor * (G @ D @ x) + delta

        # Check convergence
        if np.linalg.norm(x - x_old, 1) < tolerance:
            break

    # Calculate risk thresholds
    risk_threshold_90 = np.percentile(x, 90)  # Top 10% threshold
    risk_threshold_95 = np.percentile(x, 95)  # Top 5% threshold
    risk_threshold_99 = np.percentile(x, 99)  # Top 1% threshold

    # Identify high-risk providers
    high_risk_indices = np.where(x > risk_threshold_90)[0]
    extreme_risk_indices = np.where(x > risk_threshold_99)[0]

    # Calculate risk amplification factors
    amplification_factors = np.where(risk_scores > 0, x / risk_scores, 1.0)

    # Value at Risk calculations
    var_95 = np.percentile(x, 95)  # 95% confidence level
    var_99 = np.percentile(x, 99)  # 99% confidence level

    # Expected Shortfall calculations
    es_95 = np.mean(x[x >= var_95]) if np.any(x >= var_95) else var_95
    es_99 = np.mean(x[x >= var_99]) if np.any(x >= var_99) else var_99

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

def analyze_provider_network(G, index_to_npi=None, damping_factor=0.85, tolerance=1e-8):
    """
    Analyzes provider network structure and identifies key players.

    Categorizes providers by their network roles:
    - Hub providers: High outgoing referrals
    - Authority providers: High incoming referrals
    - Central providers: High PageRank scores

    Args:
        G (scipy.sparse.csc_matrix): Provider referral network adjacency matrix
        index_to_npi (dict): Maps matrix indices to NPI numbers
        damping_factor (float): PageRank damping parameter (default: 0.85)
        tolerance (float): PageRank convergence tolerance (default: 1e-8)

    Returns:
        dict: Network analysis results with provider categories and metrics
    """
    validate_graph(G)
    logger.info("Starting provider network analysis")

    if index_to_npi is None:
        index_to_npi = {i: i for i in range(G.shape[0])}

    # Calculate PageRank centrality scores
    centrality_scores, iterations, converged, error, cpu_time = pagerank(
        G, damping_factor=damping_factor, tolerance=tolerance
    )

    logger.info(f"PageRank analysis: {iterations} iterations, converged={converged}, time={cpu_time:.4f}s")

    # Calculate degree centralities
    in_degrees = np.array(G.sum(axis=0)).flatten()  # Incoming referrals (authority measure)
    out_degrees = np.array(G.sum(axis=1)).flatten()  # Outgoing referrals (hub measure)

    # Set thresholds for top 10% providers in each category
    hub_threshold = np.percentile(out_degrees, 90)
    authority_threshold = np.percentile(in_degrees, 90)
    central_threshold = np.percentile(centrality_scores, 90)

    # Identify key provider categories
    hub_providers = np.where(out_degrees > hub_threshold)[0]  # High referrers
    authority_providers = np.where(in_degrees > authority_threshold)[0]  # High receivers
    central_providers = np.where(centrality_scores > central_threshold)[0]  # High centrality

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

            results['key_providers'][provider_type].append(provider_info)

    logger.info(f"Provider network analysis complete. Identified {len(hub_providers)} hubs, {len(authority_providers)} authorities, {len(central_providers)} central providers")
    return results

def analyze_convergence(G, tolerance_range=None):
    """
    Tests PageRank algorithm performance across different tolerance levels.

    Args:
        G (scipy.sparse.csc_matrix): Provider referral network adjacency matrix
        tolerance_range (list): Tolerance levels to test (default: [1e-4, 1e-6, 1e-8, 1e-10, 1e-12])

    Returns:
        dict: Convergence results for each tolerance level
    """
    if tolerance_range is None:
        tolerance_range = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

    # Store results for each tolerance level
    convergence_data = {}

    # Test each tolerance level
    for tol in tolerance_range:
        try:
            start_time = time.time()

            n = G.shape[0]
            c = G.sum(axis=0).A[0]

            D = identity(n, format='csc')
            D.setdiag(1/np.where(c != 0, c, 1))

            delta = (1-0.85)/n * np.ones(n)
            x = np.ones(n) / n

            converged = False
            error = 0.0
            max_iter = 1000

            # Run PageRank with current tolerance
            for iteration in range(max_iter):
                x_old = x.copy()

                x = 0.85 * (G @ D @ x) + delta * np.sum(x)
                x = x / np.sum(x)

                error = np.linalg.norm(x - x_old, 1)

                if error < tol:
                    converged = True
                    break

            cpu_time = time.time() - start_time

            # Store results
            convergence_data[tol] = {
                'iterations': iteration + 1,
                'cpu_time': cpu_time,
                'converged': converged,
                'final_error': float(error)
            }

            logger.info(f"Tolerance {tol}: {iteration + 1} iterations, converged={converged}, error={error:.2e}")

        except Exception as e:
            logger.warning(f"Failed at tolerance {tol}: {e}")
            convergence_data[tol] = None

    return convergence_data

def load_graph(filename):
    """
    Loads provider referral network from file and creates adjacency matrix.

    Args:
        filename (str): Path to referral network data file

    Returns:
        tuple: (G, index_to_npi)
            - G (scipy.sparse.csc_matrix): Provider referral adjacency matrix
            - index_to_npi (dict): Maps matrix indices to NPI numbers
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
    Runs comprehensive healthcare network analysis.

    Args:
        G (scipy.sparse.csc_matrix): Network adjacency matrix
        index_to_npi (dict): Maps matrix indices to NPI numbers

    Returns:
        dict: Complete analysis results
    """
    logger.info(f"Starting healthcare analysis on full dataset: {G.shape[0]:,} providers")

    # Core PageRank analysis
    pagerank_result = analyze_network(G, index_to_npi)

    # Provider Network structure analysis
    provider_result = analyze_provider_network(G, index_to_npi=index_to_npi)

    # Construct risk scores
    in_degrees = np.array(G.sum(axis=0)).flatten()
    out_degrees = np.array(G.sum(axis=1)).flatten()
    total_degrees = in_degrees + out_degrees

    # Connectivity risk: higher connectivity = higher network exposure
    connectivity_risk = total_degrees / np.max(total_degrees) if np.max(total_degrees) > 0 else np.zeros(G.shape[0])

    # Referral imbalance risk: measures pattern irregularities
    referral_imbalance = np.where(in_degrees > 0, out_degrees / (in_degrees + 1e-10), out_degrees)
    referral_imbalance = referral_imbalance / np.max(referral_imbalance) if np.max(referral_imbalance) > 0 else np.zeros(G.shape[0])

    # Isolation risk: providers with fewer connections
    isolation_risk = 1.0 / (1.0 + total_degrees)
    isolation_risk = isolation_risk / np.max(isolation_risk) if np.max(isolation_risk) > 0 else np.zeros(G.shape[0])

    # Combine risk components with equal weights
    base_risk = 0.001  # Minimum risk for all providers
    risk_scores = (base_risk +
                  (1/3) * connectivity_risk +
                  (1/3) * referral_imbalance +
                  (1/3) * isolation_risk)

    # Normalize to probability distribution
    risk_scores = risk_scores / np.sum(risk_scores)

    # Risk propagation analysis
    risk_result = analyze_risk_propagation(G, risk_scores, index_to_npi)

    # Algorithm performance analysis
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
    Main execution: loads data, runs analysis, creates visualizations, displays results.
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
        print(f" - Total Providers: {metrics['total_providers']:,}")
        print(f" - Total Referrals: {metrics['total_referrals']:,}")
        print(f" - Average Referrals Out: {metrics['average_referrals_out']:.2f}")
        print(f" - Average Referrals In: {metrics['average_referrals_in']:.2f}")
        print(f" - Network Density: {metrics['network_density']:.2e}")

        print(f"\nKey Provider Categories:")
        for provider_type, providers in provider_results['key_providers'].items():
            print(f" - {provider_type.title()}: {len(providers)} providers")
            if providers:
                top_provider = providers[0]
                print(f"Top Provider: NPI {top_provider['npi']}")
                print(f"Centrality: {top_provider['centrality_score']:.4f}")
                print(f"Referrals Out/In: {top_provider['referrals_out']}/{top_provider['referrals_in']}")

        print(f"\nPageRank Analysis:")
        print(f" - Converged: {'Yes' if pagerank_result.converged else 'No'}")
        print(f" - Iterations: {pagerank_result.iterations}")
        print(f" - Computation Time: {pagerank_result.cpu_time:.4f}s")
        print(f" - Final Error: {pagerank_result.error:.2e}")

        print(f"\nRisk Analysis Results:")
        stats = risk_results['risk_statistics']
        print(f" - Mean Risk Score: {stats['mean_risk']:.6f}")
        print(f" - Maximum Risk Score: {stats['max_risk']:.6f}")
        print(f" - Risk Concentration (top 10%): {stats['risk_concentration_90']:.2%}")
        print(f" - High-risk Providers: {len(risk_results['high_risk_providers'])}")
        print(f" - VaR 95%: {stats['var_95']:.6f}")
        print(f" - Expected Shortfall 95%: {stats['expected_shortfall_95']:.6f}")

        if risk_results['high_risk_providers']:
            print(f"\nTop 3 High-Risk Providers:")
            for i, provider in enumerate(risk_results['high_risk_providers'][:3]):
                print(f"{i+1}. NPI {provider['npi']}")
                print(f"Risk Score: {provider['risk_score']:.6f}")
                print(f"Amplification: {provider['amplification_factor']:.2f}x")

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
            print(f" - Iteration Range: {min(iterations)} - {max(iterations)}")
            print(f" - Time Range: {min(times):.4f}s - {max(times):.4f}s")
            print(f" - Average Iterations: {np.mean(iterations):.1f}")

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