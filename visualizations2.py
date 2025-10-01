"""
Healthcare Network Visualization Module

Comprehensive visualization suite for healthcare PageRank analysis results.
Provides both static matplotlib plots and interactive Plotly dashboards for exploring provider network patterns and risk analysis.

Author: [Your Name]
Date: [Current Date]
License: MIT

Key Features:
- Static publication-quality matplotlib visualizations
- Interactive Plotly dashboard for data exploration
- Healthcare-specific color schemes and layouts
- Risk analysis visualizations with actuarial metrics
- Convergence analysis plots for algorithm performance
- Professional styling suitable for business presentations
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import logging
from pathlib import Path

# Plotly imports for interactive dashboards
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available")

logger = logging.getLogger(__name__)

# Create visualizations directory
VISUALIZATIONS_DIR = Path("visualizations")
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

def get_viz_path(filename):
    """
    Generate full path for visualization files.

    Args:
        filename (str): Name of the visualization file

    Returns:
        Path: Full path to the visualization file
    """
    return VISUALIZATIONS_DIR / filename

def create_healthcare_analysis_plots(G, pagerank_result, provider_results):
    """
    Create comprehensive static visualizations for healthcare PageRank analysis.

    Generates a 6-panel matplotlib figure showing key network metrics,
    PageRank distributions, provider categories, and algorithm performance.
    Optimized for publication-quality output and business presentations.

    Args:
        G (scipy.sparse.csc_matrix): Network adjacency matrix (unused but kept for API consistency)
        pagerank_result (RankResult): PageRank computation results
        provider_results (dict): Provider network analysis results

    Saves:
        healthcare_pagerank_analysis.png: 6-panel analysis visualization
    """
    try:
        values = pagerank_result.values

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Healthcare Network PageRank Analysis', fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('PageRank Values by Provider Rank', fontsize=12)

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
                color='#2980b9', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Provider Rank')
        ax1.set_ylabel('PageRank Score')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('PageRank Score Distribution', fontsize=12)

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
    Create comprehensive risk analysis visualizations for healthcare networks.

    Generates 4-panel matplotlib figure showing risk score distributions, rankings, statistics, and 
    provider categorization. Includes metrics like VaR and Expected Shortfall for financial risk assessment.

    Args:
        risk_results (dict): Risk analysis results containing:
            - propagated_risk_scores: Array of risk scores for each provider
            - risk_statistics: Dict with VaR, ES, and other risk metrics
            - high_risk_providers: List of providers exceeding risk thresholds
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

        axes[0, 0].hist(risk_scores, bins=40, alpha=0.7, color='#e74c3c', edgecolor='black')
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Risk Score Distribution')
        axes[0, 0].grid(True, alpha=0.3)

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

def create_convergence_analysis_plots(convergence_results):
    """
    Create algorithm convergence analysis visualizations.

    Generates a 2-panel matplotlib figure showing how PageRank algorithm performance varies with different 
    tolerance levels. Useful for understanding computational trade-offs and optimal parameter selection.

    Args:
        convergence_results (dict): Results from convergence analysis containing:
            - Key: tolerance level (float)
            - Value: dict with 'iterations', 'cpu_time', 'converged' fields

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

# Dashboard data storage
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
    Create an interactive Plotly dashboard for healthcare network analysis.

    Generates a comprehensive 6-panel interactive HTML dashboard combining
    PageRank analysis, risk assessment, and algorithm performance metrics.
    Includes hover tooltips, zoom capabilities, and professional styling
    suitable for business presentations and data exploration.

    Dashboard Panels:
        1. PageRank Values by Rank (log-scale line plot)
        2. PageRank Value Distribution (histogram with density)
        3. Risk Score Distribution (histogram)
        4. Risk Propagation Analysis (scatter plot)
        5. Convergence Analysis (line plot with tolerance levels)
        6. Provider Risk Categories (pie chart)

    Saves:
        healthcare_network_dashboard.html: Interactive dashboard file

    Note:
        Requires Plotly to be installed. Uses global _dashboard_data
        populated by add_to_dashboard() calls from main analysis.
    """
    global _dashboard_data
    try:
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for dashboard creation")
            return

        # Create multi-panel subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'PageRank Values by Rank', 'PageRank Value Distribution',
                'Risk Score Distribution', 'Risk Propagation Analysis',
                'Convergence Analysis', 'Provider Risk Categories'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "pie"}]
            ],
            vertical_spacing=0.15,  # Increased from 0.12 to create more space between rows
            horizontal_spacing=0.12  # Increased from 0.10 to create more space between columns
        )

        # 1. PageRank Values by Rank
        if _dashboard_data['pagerank_data']:
            for size, pr_values_dict in _dashboard_data['pagerank_data'].items():
                method_name, values = list(pr_values_dict.items())[0]

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

                # 2. PageRank Distribution
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

            fig.add_trace(
                go.Histogram(
                    x=risk_scores,
                    name='Risk Score Distribution',
                    marker=dict(color='#e74c3c'),
                    opacity=0.8,
                    nbinsx=50,
                    showlegend=False,
                    hovertemplate='<b>Risk Score Distribution</b><br>' +
                                'Risk Score: %{x:.2e}<br>' +
                                'Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )

            # 4. Risk Propagation Analysis
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
            converged_status = [data['converged'] if data else False for data in conv_data.values()]

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
                    textfont=dict(size=10),  # Smaller text to reduce space usage
                    hole=0.3,  # Add a hole in the center to make it a donut chart (saves space)
                    hovertemplate='<b>%{label}</b><br>' +
                                'Count: %{value}<br>' +
                                'Percentage: %{percent}<extra></extra>'
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Healthcare Network Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            height=1300,  # Increased height to accommodate better spacing
            showlegend=False,
            margin=dict(t=120, b=60, l=60, r=60),  # Increased margins for better spacing
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=11)
        )

        # Update axes
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

        # Save dashboard
        filename = 'healthcare_network_dashboard.html'
        filepath = get_viz_path(filename)
        fig.write_html(filepath)

        logger.info(f"Created comprehensive dashboard: {filepath}")

        # Reset dashboard data
        _dashboard_data = {
            'pagerank_data': {},
            'risk_data': None,
            'performance_data': {},
            'convergence_data': {}
        }

    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
