# Healthcare Network PageRank Analysis

Identifying influential healthcare providers and modeling systemic risk propagation in a **1M+ node referral network** using scalable PageRank algorithms.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Sparse-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## TL;DR
- Applied **PageRank to a 1M+ node healthcare referral network**
- Identified **top 10% of providers responsible for ~54% of propagated risk**
- Demonstrated **network-driven risk amplification up to 400x**
- Built a **memory-efficient sparse implementation** scalable to real CMS data

---

## Project Overview

This project applies the **PageRank algorithm** to a large-scale healthcare provider referral network to identify influential providers and analyze how risk propagates through referral relationships.

Using the **2015 CMS Physician Shared Patient Patterns dataset**, the network contains:
- **1,034,612** healthcare providers
- **65,739,582** referral relationships.

The project emphasizes:
- Scalability to million-node graphs
- Numerical stability of eigenvector computation
- Domain-specific interpretation of network centrality in healthcare

---

## From Coursework to Large-Scale Healthcare Analysis
This project extends a PageRank implementation originally developed for **MAT 167 (Applied Linear Algebra)**.

### Key Extensions
- Scaled from **~280K web nodes to 1M+ healthcare providers**
- Replaced dense matrices with **SciPy sparse representations**
- Implemented **power iteration** for large-scale convergence
- Reframed PageRank as a **risk propagation model**
- Added convergence benchmarking and performance analysis

[Original PageRank Analysis (MAT 167)](original_pagerank/original_pagerank_report.pdf)

---

## Key Findings

### Network Structure
- **Highly centralized network:** Top 10% of providers account for 54% of network influence
- **Power-law distribution:** Small number of providers have disproportionate connectivity
- **Average connectivity:** 63.54 referrals per provider across 1M+ nodes

### Risk Propagation
- **Risk concentration:** 54.21% of total risk concentrated in top 10% of providers
- **Network amplification:** Provider network position can amplify baseline risk by up to 400x
- **Intervention leverage:** Targeting the most central 10% of providers could impact majority of network-level risk

### Algorithm Performance
- **Convergence speed:** 72 iterations (L1 tolerance = 1e-8)
- **Memory efficiency:** ~2 GB
- **Computation time:** ~39 mins from start to finish

---

## Dataset

**Source:** [CMS Physician Shared Patient Patterns (2015)]([https://data.cms.gov/](https://www.nber.org/research/data/physician-shared-patient-patterns-data))
- **Providers:** 1,034,612
- **Edges:** 65,739,582
- **Network Density:** 6.14 × 10⁻⁵ (highly sparse)
- **Average Degree:** 63.54 referrals per provider

Data contains no patient-level information and uses anonymized provider identifiers.

---

## Technical Implementation

### Algorithm: PageRank with Power Iteration
```
x_new = damping_factor × (G @ D @ x_old) + (1-damping_factor)/n × 1
```

Where:
- `G` = Adjacency matrix (who refers to whom)
- `D` = Normalization matrix (divides by out-degree)
- `x` = PageRank vector (probability distribution)
- `damping_factor` = 0.85 (probability of following links vs random jump)

### Key Technical Decisions

#### 1. **Sparse Matrix Optimization**
- Dense matrix would require ~8 TB of memory
- Implemented using `scipy.sparse.csc_matrix`
- Reduced memory footprint by **~4000x**

#### 2. **Convergence Criterion**
- **L1 norm** for probability stability
- Tolerance: `1e-8` (optimal accuracy/runtime tradeoff)

#### 3. **Risk Propagation Model**
- Higher damping factor (0.95)
- Non-uniform initialization using composite provider risk
- Amplification factors quantify network effects

### Risk Propagation Model

Adapted PageRank to model risk flow:
- **Higher damping factor (0.95)** to model persistent risk retention
- **Non-uniform initialization** using composite provider risk scores
- **Amplification factors** quantify how network position magnifies baseline risk
- **Top-percentile risk concentration analysis (90/95/99%)**

This approach identifies providers whose network position drives risk, even when their baseline risk is low.

---

## Risk Propagation Results

| Rank | NPI | Risk Score | Amplification Factor |
|------|-----|------------|---------------------|
| 1 | 1538144910 | 0.001943 | **397.82x** |
| 2 | 1063497451 | 0.001307 | **398.57x** |
| 3 | 1700865094 | 0.001187 | **222.96x** |

---

## Methodology

This project implements a scalable, end-to-end pipeline for analyzing large healthcare referral networks using PageRank-based centrality and risk propagation modeling.

The methodology is structured into five analytical stages, separating data engineering, numerical computation, network analysis, and risk interpretation.

1. **Data Ingestion & Graph Construction**
   - Loaded CMS referral edge lists from CSV format
   - Mapped provider NPIs to contiguous matrix indices for efficient computation
   - Constructed a **directed referral graph** where edges represent shared-patient referrals
   - Stored adjacency information using **SciPy sparse matrices** to accommodate extreme sparsity
  
   Explicit graph construction enables linear algebra-based centrality analysis while remaining memory-efficient at million-node scale

2. **PageRank Centrality Computation**
   - Implemented **PageRank via power iteration**
   - Used sparse matrix-vector multiplication for scalability
   - Applied dangling-node correction to preserve probability mass
   - Monitored convergence using the **L1 norm**, appropriate for probability distributions

   **Key Parameters**
   - Damping factor: 0.85
   - Convergence tolerance: 1e-8
   - Typical convergence: 50-100 iterations

   Power iteration provides a numerically stable and interpretable method for estimating eigenvector centrality in massive directed networks.

3. **Network Structure Analysis**
   Computed multiple complementary measures to characterize provider roles:
   - **In-degree (Authorities):** High incoming referral volume
   - **Out-degree (Hubs):** High outgoing referral volume
   - **PageRank Centrality:** Global network influence

   Using multiple centrality metrics avoids over-reliance on a single notion of "importance" and captures different referral dynamics.

4. **Risk Modeling & Propagation**
   PageRank was adapted into a **risk diffusion framework**:
   - Initialized the PageRank vector with **composite provider risk scores**
   - Increased damping factor (0.95) to model persistent risk retention
   - Interpreted steady-state PageRank values as **network-amplified risk**
   - Computed **amplification factors** to quantify how network position magnifies baseline risk

   Risk concentration was evaluated using:
   - Top-percentile risk shares (90/95/99%)
   - Value at Risk (VaR)
   - Expected Shortfall (ES)

   This approach isolates risk driven by network structure rather than individual provider attributes alone.

5. **Performance & Convergence Testing**
   - Benchmarked convergence behavior across multiple tolerance levels
   - Evaluated runtime and iteration counts for scalability assessment
   - Identified diminishing returns beyond tolerance = 1e-8
  
   Explicit performance evaluation ensures the methodology is practical for real-world healthcare-scale networks.

---

## Reproducibility

### Environment
- Python 3.7+
- SciPy sparse linear algebra
- Tested on commodity hardware (>=32GP RAM recommended)

### Run
```
pip install numpy scipy pandas matplotlib plotly networkx
python pagerank.py
```

Outputs include:
- Static plots
- Interactive HTML dashboard
- Execution logs

---

## Project Structure

```
pagerank-shared-patterns/
│
├── pagerank.py              # Core PageRank algorithm implementation
├── visualizations.py        # Visualization generation module
├── healthcare_network_dashboard.html  # Generated dashboard
├── visualizations/           # Generated plots
│   ├── healthcare_pagerank_analysis.png
│   ├── healthcare_risk_analysis.png
│   └── convergence_analysis.png
├── original_pagerank/        # Original PageRank project this project is based on
│   ├── original_pagerank_report.pdf  # Project report
│   ├── pagerank.py           # Original PageRank implementation
│   ├── visualizations.py     # Original visualization generation
│   └── web_Stanford.txt      # Dataset for original PageRank project
├── pagerank_analysis.log     # Execution logs
└── README.md                 # This file
```

---

## Ethical & Practical Considerations
- This analysis **does not evaluate provider quality**
- High centrality **does not equal** poor performance
- Results should be used for **system-level monitoring**, not punitive action
- Demonstrates how network structure can bias risk attribution

---

## Technologies Used

- **Python 3.7+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **SciPy (sparse)**: Sparse matrix operations for memory efficiency
- **Matplotlib**: Static publication-quality visualizations
- **Plotly**: Interactive HTML dashboards
- **NetworkX**: Network topology visualization (optional)
- **Pandas**: Data manipulation and correlation analysis

### Numerical Methods
- Power iteration for eigenvalue computation
- L1-norm convergence for probability distributions
- Sparse linear algebra for scalability
- Value at Risk (VaR) and Expected Shortfall for tail-risk analysis

### Network Science
- PageRank centrality
- Hub and authority scores
- Risk diffusion modeling
- Network inequality metrics (Gini coefficient)

---

## Author

**Angelina Cottone**
B.S. Statistics (Statistical Data Science), UC Davis 2025

---
*Last Updated: January 2026*
