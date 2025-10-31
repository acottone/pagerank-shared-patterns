"""
Healthcare Network Visualization

This module contains all the code for visualizing the results of the PageRank analysis on healthcare networks.

Author: Angelina Cottone
Date: August 2025
License: MIT
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import logging
from pathlib import Path

# Try to import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not installed")

logger = logging.getLogger(__name__)

# Directory to save visualizations
VISUALIZATIONS_DIR = Path("visualizations")
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

HEALTHCARE_COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'success': '#27ae60',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'light': '#ecf0f1',
    'dark': '#34495e' 
}

# Professional color palette for multi-category plots
CATEGORY_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

def get_viz_path(filename):
    """
    Makes the full path to save visualization files.

    Args:
        filename (str): What to name the file

    Returns:
        Path: The full path where to save it
    """
    return VISUALIZATIONS_DIR / filename

def create_network_topology_plot(G, pagerank_values, index_to_npi=None, max_nodes=100):
    """
    Creates a network topology visualization showing provider connections and PageRank importance.

    Args:
        G (scipy.sparse.csc_matrix): The network adjacency matrix
        pagerank_values (np.ndarray): PageRank scores for each provider
        index_to_npi (dict, optional): Mapping from indices to provider NPIs
        max_nodes (int): Maximum number of nodes to display (default: 100)

    Saves:
        network_topology.png: Network visualization with node sizes based on PageRank scores
    """
    try:
        import networkx as nx

        # Sample the network for visualization if too large
        if G.shape[0] > max_nodes:
            # Get top PageRank providers and some random ones
            top_indices = np.argsort(pagerank_values)[-max_nodes//2:]
            remaining_indices = np.argsort(pagerank_values)[:-max_nodes//2]
            random_indices = np.random.choice(remaining_indices, max_nodes//2, replace=False)
            selected_indices = np.concatenate([top_indices, random_indices])

            # Create subgraph
            subgraph = G[selected_indices][:, selected_indices]
            sub_pagerank = pagerank_values[selected_indices]
        else:
            subgraph = G
            sub_pagerank = pagerank_values
            selected_indices = np.arange(G.shape[0])

        # Convert to NetworkX graph
        nx_graph = nx.from_scipy_sparse_array(subgraph, create_using=nx.DiGraph)

        # Create layout
        pos = nx.spring_layout(nx_graph, k=1, iterations=50)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Node sizes based on PageRank scores
        node_sizes = (sub_pagerank / np.max(sub_pagerank)) * 1000 + 50

        # Node colors based on PageRank quartiles
        quartiles = np.percentile(sub_pagerank, [25, 50, 75])
        node_colors = []
        for score in sub_pagerank:
            if score >= quartiles[2]:
                node_colors.append(HEALTHCARE_COLORS['accent'])  # Top quartile - red
            elif score >= quartiles[1]:
                node_colors.append(HEALTHCARE_COLORS['warning'])  # Third quartile - orange
            elif score >= quartiles[0]:
                node_colors.append(HEALTHCARE_COLORS['secondary'])  # Second quartile - blue
            else:
                node_colors.append(HEALTHCARE_COLORS['light'])  # Bottom quartile - light gray

        # Draw network
        nx.draw_networkx_nodes(nx_graph, pos, node_size=node_sizes, node_color=node_colors,
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, edge_color=HEALTHCARE_COLORS['dark'],
                              arrows=True, arrowsize=10, ax=ax)

        # Add title and legend
        ax.set_title('Healthcare Provider Network Topology\n(Node size = PageRank importance)',
                    fontsize=14, fontweight='bold', pad=20)

        # Create legend
        legend_elements = [
            plt.scatter([], [], s=100, c=HEALTHCARE_COLORS['accent'], alpha=0.8, label='Top 25% (High Influence)'),
            plt.scatter([], [], s=100, c=HEALTHCARE_COLORS['warning'], alpha=0.8, label='75th-50th percentile'),
            plt.scatter([], [], s=100, c=HEALTHCARE_COLORS['secondary'], alpha=0.8, label='50th-25th percentile'),
            plt.scatter([], [], s=100, c=HEALTHCARE_COLORS['light'], alpha=0.8, label='Bottom 25%')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        ax.set_axis_off()
        plt.tight_layout()

        filepath = get_viz_path("network_topology.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved network topology visualization: {filepath}")

    except ImportError:
        logger.warning("NetworkX not available - skipping network topology plot")
    except Exception as e:
        logger.error(f"Network topology visualization failed: {e}")

def create_healthcare_analysis_plots(G, pagerank_result, provider_results):
    """
    Makes a big 6-panel plot showing all the important PageRank results.

    Args:
        G (scipy.sparse.csc_matrix): The network matrix (not actually used here)
        pagerank_result (PageRankResult): All the PageRank results from our analysis
        provider_results (dict): Info about different types of providers

    Saves:
        healthcare_pagerank_analysis.png: A PNG file in the visualizations folder
    """
    try:
        values = pagerank_result.values

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Healthcare Network PageRank Analysis', fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('PageRank Values by Provider Rank', fontsize=12)

        # Sample data for large networks to improve performance
        if len(values) > 200:
            top_indices = np.argsort(values)[-100:]
            remaining_indices = np.argsort(values)[:-100]
            step = max(1, len(remaining_indices) // 100)
            spaced_indices = remaining_indices[::step][:100]
            sample_indices = np.concatenate([top_indices, spaced_indices])
            sample_values = values[sample_indices]
        else:
            sample_values = values

        sorted_values = sorted(sample_values, reverse=True)
        ax1.plot(range(1, len(sorted_values) + 1), sorted_values,
                color=HEALTHCARE_COLORS['secondary'], linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Provider Rank')
        ax1.set_ylabel('PageRank Score')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('PageRank Score Distribution', fontsize=12)

        # Filter outliers for better visualization
        q99 = np.percentile(values, 99)
        q1 = np.percentile(values, 1)
        filtered_values = values[(values >= q1) & (values <= q99)]

        ax2.hist(filtered_values, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax2.set_xlabel('PageRank Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Network Overview', fontsize=12)

        metrics = provider_results['network_metrics']
        overview_labels = ['Providers', 'Referrals', 'Avg Out', 'Avg In']
        overview_values = [
            metrics['total_providers'],
            metrics['total_referrals'],
            metrics['average_referrals_out'],
            metrics['average_referrals_in']
        ]

        bars = ax3.bar(overview_labels, overview_values,
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
        ax3.set_ylabel('Count/Average')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, overview_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}' if value > 1 else f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('Top 10 Providers by PageRank', fontsize=12)

        top_10_indices = np.argsort(values)[-10:]
        top_10_values = values[top_10_indices]

        ax4.barh(range(10), top_10_values, color='#2980b9', alpha=0.8)
        ax4.set_xlabel('PageRank Score')
        ax4.set_ylabel('Provider Rank')
        ax4.set_yticks(range(10))
        ax4.set_yticklabels([f'#{i+1}' for i in range(10)])
        ax4.grid(True, alpha=0.3, axis='x')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('Key Provider Categories', fontsize=12)

        categories = list(provider_results['key_providers'].keys())
        category_counts = [len(providers) for providers in provider_results['key_providers'].values()]

        ax5.pie(category_counts, labels=categories, autopct='%1.1f%%',
                colors=['#e74c3c', '#3498db', '#2ecc71'], startangle=90)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Algorithm Performance', fontsize=12)

        perf_labels = ['Iterations', 'Time (ms)', 'Converged']
        perf_values = [
            pagerank_result.iterations,
            pagerank_result.cpu_time * 1000,
            1 if pagerank_result.converged else 0
        ]

        bars = ax6.bar(perf_labels, perf_values,
                      color=['#9b59b6', '#f39c12', '#2ecc71' if pagerank_result.converged else '#e74c3c'],
                      alpha=0.8)
        ax6.set_ylabel('Value')

        for bar, value in zip(bars, perf_values):
            height = bar.get_height()
            if bar == bars[2]:
                label = 'Yes' if value == 1 else 'No'
            else:
                label = f'{value:.0f}'
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filename = 'healthcare_pagerank_analysis.png'
        filepath = get_viz_path(filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved healthcare analysis visualization: {filepath}")

    except Exception as e:
        logger.error(f"Healthcare analysis visualization failed: {e}")

def create_risk_analysis_plots(risk_results, title="Healthcare Risk Analysis"):
    """
    Creates comprehensive risk analysis visualizations for healthcare networks.

    Generates 4-panel matplotlib figure with risk distributions, rankings, statistics,
    and provider categories. Includes VaR and Expected Shortfall metrics.

    Args:
        risk_results (dict): Risk analysis results with scores, statistics, and high-risk providers
        title (str): Title for the overall figure

    Saves:
        healthcare_risk_analysis.png: 4-panel risk analysis visualization
    """
    try:
        if 'propagated_risk_scores' not in risk_results:
            logger.warning("No risk scores available for plotting")
            return

        risk_scores = risk_results['propagated_risk_scores']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        axes[0, 0].hist(risk_scores, bins=40, alpha=0.7, color=HEALTHCARE_COLORS['accent'], edgecolor='black')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Risk Score Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Add statistical annotations
        mean_risk = np.mean(risk_scores)
        median_risk = np.median(risk_scores)
        axes[0, 0].axvline(mean_risk, color=HEALTHCARE_COLORS['primary'], linestyle='--', alpha=0.8, label=f'Mean: {mean_risk:.6f}')
        axes[0, 0].axvline(median_risk, color=HEALTHCARE_COLORS['warning'], linestyle='--', alpha=0.8, label=f'Median: {median_risk:.6f}')
        axes[0, 0].legend()

        sorted_risks = np.sort(risk_scores)[::-1]
        axes[0, 1].plot(range(len(sorted_risks)), sorted_risks, 'r-', alpha=0.7, linewidth=2)
        axes[0, 1].set_xlabel('Provider Rank')
        axes[0, 1].set_ylabel('Risk Score')
        axes[0, 1].set_title('Risk Score by Provider Rank')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        if 'risk_statistics' in risk_results:
            stats = risk_results['risk_statistics']
            key_stats = {
                'Mean Risk': stats['mean_risk'],
                'Max Risk': stats['max_risk'],
                'VaR 95%': stats['var_95'],
                'ES 95%': stats['expected_shortfall_95']
            }

            stat_names = list(key_stats.keys())
            stat_values = list(key_stats.values())

            bars = axes[1, 0].bar(stat_names, stat_values, alpha=0.7, color='#f39c12')
            axes[1, 0].set_ylabel('Risk Value')
            axes[1, 0].set_title('Key Risk Statistics')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2e}', ha='center', va='bottom', fontsize=9)

        high_risk_count = len(risk_results.get('high_risk_providers', []))
        total_providers = len(risk_scores)
        low_risk_count = total_providers - high_risk_count

        axes[1, 1].pie([high_risk_count, low_risk_count],
                      labels=['High Risk', 'Low Risk'],
                      colors=['#e74c3c', '#2ecc71'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title('Provider Risk Categories')

        plt.tight_layout()
        filename = 'healthcare_risk_analysis.png'
        filepath = get_viz_path(filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved risk analysis plot: {filepath}")

    except Exception as e:
        logger.error(f"Risk analysis plot creation failed: {e}")

def create_risk_correlation_heatmap(risk_results, pagerank_result):
    """
    Creates a correlation heatmap between risk metrics and network centrality measures.

    Args:
        risk_results (dict): Risk analysis results
        pagerank_result (PageRankResult): PageRank computation results

    Saves:
        risk_correlation_heatmap.png: Correlation matrix visualization
    """
    try:
        risk_scores = risk_results['propagated_risk_scores']
        pagerank_scores = pagerank_result.values

        # Calculate additional centrality measures if possible
        from scipy.sparse import csc_matrix
        import pandas as pd

        # Create correlation matrix
        data = {
            'Risk Score': risk_scores,
            'PageRank Score': pagerank_scores,
        }

        df = pd.DataFrame(data)
        correlation_matrix = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.index)

        # Add correlation values as text
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Risk-Centrality Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        filepath = get_viz_path("risk_correlation_heatmap.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved risk correlation heatmap: {filepath}")

    except ImportError:
        logger.warning("Pandas not available - skipping correlation heatmap")
    except Exception as e:
        logger.error(f"Risk correlation heatmap creation failed: {e}")

def create_summary_statistics_plot(pagerank_result, risk_results, provider_results):
    """
    Creates a comprehensive summary statistics visualization.

    Args:
        pagerank_result (PageRankResult): PageRank computation results
        risk_results (dict): Risk analysis results
        provider_results (dict): Provider network analysis results

    Saves:
        summary_statistics.png: Summary statistics dashboard
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle('Healthcare Network Analysis - Summary Dashboard', fontsize=18, fontweight='bold')

        # 1. Network Overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = provider_results['network_metrics']
        network_data = [
            metrics['total_providers'],
            metrics['total_referrals'],
            int(metrics['average_referrals_out']),
            int(metrics['average_referrals_in'])
        ]
        network_labels = ['Providers', 'Referrals', 'Avg Out', 'Avg In']

        bars = ax1.bar(network_labels, network_data, color=CATEGORY_COLORS[:4], alpha=0.8)
        ax1.set_title('Network Overview', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, network_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom', fontsize=9)

        # 2. PageRank Performance (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        perf_labels = ['Iterations', 'Time (ms)', 'Error (log10)']
        perf_values = [
            pagerank_result.iterations,
            pagerank_result.cpu_time * 1000,
            np.log10(pagerank_result.error) if pagerank_result.error > 0 else -10
        ]

        bars = ax2.bar(perf_labels, perf_values,
                      color=[HEALTHCARE_COLORS['info'], HEALTHCARE_COLORS['warning'], HEALTHCARE_COLORS['secondary']],
                      alpha=0.8)
        ax2.set_title('Algorithm Performance', fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Risk Statistics (top-right)
        ax3 = fig.add_subplot(gs[0, 2:])
        if 'risk_statistics' in risk_results:
            stats = risk_results['risk_statistics']
            risk_labels = ['Mean Risk', 'Max Risk', 'VaR 95%', 'ES 95%', 'Gini Coeff']
            risk_values = [
                stats['mean_risk'] * 1000,  # Scale for visibility
                stats['max_risk'] * 1000,
                stats['var_95'] * 1000,
                stats['expected_shortfall_95'] * 1000,
                stats['gini_coefficient']
            ]

            bars = ax3.bar(risk_labels, risk_values, color=HEALTHCARE_COLORS['accent'], alpha=0.8)
            ax3.set_title('Risk Metrics (×1000 for Mean, Max, VaR, ES)', fontweight='bold')
            ax3.set_ylabel('Value')
            ax3.tick_params(axis='x', rotation=45)

        # 4. PageRank Distribution (middle-left)
        ax4 = fig.add_subplot(gs[1, :2])
        pagerank_values = pagerank_result.values
        ax4.hist(pagerank_values, bins=50, alpha=0.7, color=HEALTHCARE_COLORS['secondary'], edgecolor='black')
        ax4.set_xlabel('PageRank Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('PageRank Score Distribution', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # 5. Risk Distribution (middle-right)
        ax5 = fig.add_subplot(gs[1, 2:])
        if 'propagated_risk_scores' in risk_results:
            risk_scores = risk_results['propagated_risk_scores']
            ax5.hist(risk_scores, bins=50, alpha=0.7, color=HEALTHCARE_COLORS['accent'], edgecolor='black')
            ax5.set_xlabel('Risk Score')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Risk Score Distribution', fontweight='bold')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)

        # 6. Top Providers Comparison (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        top_n = 15
        top_indices = np.argsort(pagerank_values)[-top_n:]
        top_pagerank = pagerank_values[top_indices]

        if 'propagated_risk_scores' in risk_results:
            top_risk = risk_results['propagated_risk_scores'][top_indices]

            x = np.arange(top_n)
            width = 0.35

            bars1 = ax6.bar(x - width/2, top_pagerank * 1000, width,
                           label='PageRank (×1000)', color=HEALTHCARE_COLORS['secondary'], alpha=0.8)
            bars2 = ax6.bar(x + width/2, top_risk * 1000, width,
                           label='Risk Score (×1000)', color=HEALTHCARE_COLORS['accent'], alpha=0.8)

            ax6.set_xlabel('Top Providers (by PageRank)')
            ax6.set_ylabel('Score (×1000)')
            ax6.set_title('Top 15 Providers: PageRank vs Risk Scores', fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels([f'#{i+1}' for i in range(top_n)])
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        filepath = get_viz_path("summary_statistics.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary statistics plot: {filepath}")

    except Exception as e:
        logger.error(f"Summary statistics plot creation failed: {e}")

def create_enhanced_visualizations(G, pagerank_result, provider_results, risk_results):
    """
    Creates all enhanced visualizations for comprehensive healthcare network analysis.

    Args:
        G (scipy.sparse.csc_matrix): The network adjacency matrix
        pagerank_result (PageRankResult): PageRank computation results
        provider_results (dict): Provider network analysis results
        risk_results (dict): Risk analysis results

    Creates:
        Multiple enhanced visualization files in the visualizations directory
    """
    logger.info("Creating enhanced visualization suite...")

    try:
        # 1. Original healthcare analysis plots (enhanced)
        create_healthcare_analysis_plots(G, pagerank_result, provider_results)

        # 2. Enhanced risk analysis plots
        create_risk_analysis_plots(risk_results, "Enhanced Healthcare Risk Analysis")

        # 3. Network topology visualization
        create_network_topology_plot(G, pagerank_result.values)

        # 4. Risk correlation heatmap
        create_risk_correlation_heatmap(risk_results, pagerank_result)

        # 5. Comprehensive summary dashboard
        create_summary_statistics_plot(pagerank_result, risk_results, provider_results)

        logger.info("Enhanced visualization suite completed successfully!")

    except Exception as e:
        logger.error(f"Enhanced visualization creation failed: {e}")
        raise

def create_convergence_analysis_plots(convergence_results):
    """
    Creates algorithm convergence analysis visualizations.

    Generates a 2-panel matplotlib figure showing PageRank performance across
    different tolerance levels for computational trade-off analysis.

    Args:
        convergence_results (dict): Convergence analysis results with tolerance levels
            as keys and performance metrics as values

    Saves:
        convergence_analysis.png: 2-panel convergence performance visualization
    """
    try:
        tolerances = list(convergence_results.keys())
        iterations = [data['iterations'] if data and data['converged'] else 0 for data in convergence_results.values()]
        times = [data['cpu_time'] if data and data['converged'] else 0 for data in convergence_results.values()]
        converged_status = [data['converged'] if data else False for data in convergence_results.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('PageRank Convergence Analysis', fontsize=16, fontweight='bold')

        ax1.semilogx(tolerances, iterations, 'bo-', linewidth=3, markersize=8,
                     markerfacecolor='#3498db', markeredgecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Tolerance Level')
        ax1.set_ylabel('Iterations to Convergence')
        ax1.set_title('Convergence Iterations vs Tolerance')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()

        # Mark failed convergence points
        for i, (tol, converged) in enumerate(zip(tolerances, converged_status)):
            if not converged:
                ax1.annotate('Failed', (tol, iterations[i]),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           color='red', fontweight='bold')

        ax2.semilogx(tolerances, times, 'ro-', linewidth=3, markersize=8,
                     markerfacecolor='#e74c3c', markeredgecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Tolerance Level')
        ax2.set_ylabel('CPU Time (seconds)')
        ax2.set_title('Computation Time vs Tolerance')
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        plt.tight_layout()
        filename = 'convergence_analysis.png'
        filepath = get_viz_path(filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created convergence analysis plot: {filepath}")

    except Exception as e:
        logger.error(f"Convergence visualization failed: {e}")

# Global storage for dashboard data
_dashboard_data = {
    'pagerank_data': {},
    'risk_data': None,
    'performance_data': {},
    'convergence_data': {}
}

def add_to_dashboard(data_type, data, **kwargs):
    global _dashboard_data
    if data_type == 'pagerank':
        _dashboard_data['pagerank_data'][kwargs.get('size', 'default')] = data
    elif data_type == 'risk':
        _dashboard_data['risk_data'] = data
    elif data_type == 'performance':
        _dashboard_data['performance_data'].update(data)
    elif data_type == 'convergence':
        _dashboard_data['convergence_data'] = data

def create_interactive_plots(pr_values_dict, size):
    add_to_dashboard('pagerank', pr_values_dict, size=size)

def create_comprehensive_dashboard():
    """
    Creates an interactive Plotly dashboard for healthcare network analysis.

    Generates a 6-panel interactive HTML dashboard with PageRank analysis,
    risk assessment, and algorithm performance metrics. Includes hover tooltips
    and zoom capabilities.

    Dashboard Panels:
        1. PageRank Values by Rank (log-scale line plot)
        2. PageRank Value Distribution (histogram with density)
        3. Risk Score Distribution (histogram)
        4. Risk Propagation Analysis (scatter plot)
        5. Convergence Analysis (line plot)
        6. Provider Risk Categories (pie chart)

    Saves:
        healthcare_network_dashboard.html: Interactive dashboard file

    Note:
        Requires Plotly. Uses global _dashboard_data from add_to_dashboard() calls.
    """
    global _dashboard_data
    try:
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for dashboard creation")
            return

        # Create 6-panel subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'PageRank Values by Rank', 'PageRank Value Distribution',
                'Risk Score Distribution', 'Risk Propagation Analysis',
                'Convergence Analysis', ''  # Empty title for pie chart to avoid overlap
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "pie"}]
            ],
            vertical_spacing=0.18,  # Increased spacing
            horizontal_spacing=0.12
        )

        # 1. PageRank Values by Rank
        if _dashboard_data['pagerank_data']:
            for _, pr_values_dict in _dashboard_data['pagerank_data'].items():
                _, values = list(pr_values_dict.items())[0]

                # Sample data for large networks
                if len(values) > 200:
                    top_indices = np.argsort(values)[-100:]
                    remaining_indices = np.argsort(values)[:-100]
                    step = max(1, len(remaining_indices) // 100)
                    spaced_indices = remaining_indices[::step][:100]
                    sample_indices = np.concatenate([top_indices, spaced_indices])
                    sample_values = values[sample_indices]
                else:
                    sample_values = values

                sorted_values = sorted(sample_values, reverse=True)
                rank_indices = list(range(1, len(sorted_values) + 1))

                fig.add_trace(
                    go.Scatter(
                        x=rank_indices,
                        y=sorted_values,
                        mode='lines+markers',
                        name='Healthcare PageRank',
                        line=dict(color='#3498db', width=3),
                        marker=dict(color='#2980b9', size=6),
                        showlegend=False,
                        hovertemplate='<b>Healthcare PageRank</b><br>' +
                                    'Rank: %{x}<br>' +
                                    'PageRank: %{y:.6f}<br>' +
                                    'Network Size: ' + f'{len(values):,}' + '<extra></extra>'
                    ),
                    row=1, col=1
                )

                fig.update_yaxes(type="log", row=1, col=1)

                # 2. PageRank Distribution (filter outliers)
                q99 = np.percentile(values, 99)
                q1 = np.percentile(values, 1)
                filtered_values = values[(values >= q1) & (values <= q99)]

                fig.add_trace(
                    go.Histogram(
                        x=filtered_values,
                        name='PageRank Distribution',
                        marker=dict(color='#3498db'),
                        opacity=0.8,
                        nbinsx=40,
                        showlegend=False,
                        histnorm='probability density',
                        hovertemplate='<b>PageRank Distribution</b><br>' +
                                    'PageRank Range: %{x:.6f}<br>' +
                                    'Density: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=2
                )

        # 3. Risk Score Distribution
        if _dashboard_data['risk_data']:
            risk_data = _dashboard_data['risk_data']
            risk_scores = risk_data['propagated_risk_scores']

            # Filter out zero values and use better binning for small values
            non_zero_scores = risk_scores[risk_scores > 0]
            if len(non_zero_scores) > 0:

                fig.add_trace(
                    go.Histogram(
                        x=non_zero_scores,
                        name='Risk Score Distribution',
                        marker=dict(color='#e74c3c'),
                        opacity=0.8,
                        nbinsx=30,
                        showlegend=False,
                        hovertemplate='<b>Risk Score Distribution</b><br>' +
                                    'Risk Score: %{x:.6f}<br>' +
                                    'Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
            else:
                # Fallback if all scores are zero
                fig.add_trace(
                    go.Histogram(
                        x=risk_scores,
                        name='Risk Score Distribution',
                        marker=dict(color='#e74c3c'),
                        opacity=0.8,
                        nbinsx=30,
                        showlegend=False,
                        hovertemplate='<b>Risk Score Distribution</b><br>' +
                                    'Risk Score: %{x:.6f}<br>' +
                                    'Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )

            # 4. Risk Propagation Analysis (top 30 high-risk providers)
            if risk_data['high_risk_providers']:
                high_risk_scores = [p['risk_score'] for p in risk_data['high_risk_providers'][:30]]
                original_scores = [p['original_risk'] for p in risk_data['high_risk_providers'][:30]]
                provider_ids = [f"NPI {str(p['npi'])[-4:]}" for p in risk_data['high_risk_providers'][:30]]

                fig.add_trace(
                    go.Scatter(
                        x=original_scores,
                        y=high_risk_scores,
                        mode='markers',
                        name='Risk Propagation',
                        marker=dict(color='#f39c12', size=8),
                        opacity=0.8,
                        showlegend=False,
                        text=provider_ids,
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Original Risk: %{x:.6f}<br>' +
                                    'Propagated Risk: %{y:.6f}<extra></extra>'
                    ),
                    row=2, col=2
                )

        # 5. Convergence Analysis
        if _dashboard_data['convergence_data']:
            conv_data = _dashboard_data['convergence_data']
            tolerances = list(conv_data.keys())
            iterations = [data['iterations'] if data and data['converged'] else 0 for data in conv_data.values()]

            fig.add_trace(
                go.Scatter(
                    x=tolerances,
                    y=iterations,
                    mode='lines+markers',
                    name='Convergence Iterations',
                    line=dict(color='#3498db', width=4),
                    marker=dict(size=8, color='#3498db'),
                    showlegend=False,
                    hovertemplate='<b>Convergence Analysis</b><br>' +
                                'Tolerance: %{x:.0e}<br>' +
                                'Iterations: %{y}<extra></extra>'
                ),
                row=3, col=1
            )

        # 6. Provider Risk Categories
        if _dashboard_data['risk_data']:
            risk_data = _dashboard_data['risk_data']
            high_risk_count = len(risk_data['high_risk_providers'])
            total_providers = len(risk_data['propagated_risk_scores'])
            low_risk_count = total_providers - high_risk_count

            fig.add_trace(
                go.Pie(
                    labels=['High Risk Providers', 'Low Risk Providers'],
                    values=[high_risk_count, low_risk_count],
                    name="Risk Categories",
                    marker=dict(colors=['#e74c3c', '#2ecc71']),
                    textinfo='label+percent+value',
                    textfont=dict(size=10),
                    hole=0.3,  # Donut chart style
                    hovertemplate='<b>%{label}</b><br>' +
                                'Count: %{value}<br>' +
                                'Percentage: %{percent}<extra></extra>'
                ),
                row=3, col=2
            )

        # Add title annotation for pie chart (to avoid overlap)
        fig.add_annotation(
            text="Provider Risk Categories",
            x=0.825,  # Position over the pie chart (right column)
            y=0.35,   # Position above the pie chart
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, family='Arial, sans-serif'),
            xanchor="center"
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Healthcare Network Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            height=1300,
            showlegend=False,
            margin=dict(t=120, b=60, l=60, r=60),
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=11)
        )

        # Configure axes labels and styling
        fig.update_xaxes(title_text="Provider Rank", row=1, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_yaxes(title_text="PageRank Value", row=1, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_xaxes(title_text="PageRank Value", row=1, col=2, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_yaxes(title_text="Density", row=1, col=2, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_xaxes(title_text="Risk Score", row=2, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_yaxes(title_text="Frequency", row=2, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_xaxes(title_text="Original Risk Score", row=2, col=2, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_yaxes(title_text="Propagated Risk Score", row=2, col=2, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_xaxes(title_text="Tolerance Level", type="log", row=3, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)
        fig.update_yaxes(title_text="Iterations", row=3, col=1, gridcolor='rgba(0,0,0,0.1)', showgrid=True)

        # Save interactive dashboard
        filename = 'healthcare_network_dashboard.html'
        filepath = get_viz_path(filename)
        fig.write_html(filepath)

        logger.info(f"Created comprehensive dashboard: {filepath}")

        # Clear dashboard data for next use
        _dashboard_data = {
            'pagerank_data': {},
            'risk_data': None,
            'performance_data': {},
            'convergence_data': {}
        }

    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")