# Healthcare Network PageRank Analysis

Identifying influential healthcare providers and modeling risk propagation through a million-node referral network using PageRank algorithms.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Sparse-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## TL;DR
- Applied PageRank to a 1M+ node healthcare referral network
- Identified high-impact providers responsible for 54% of propagated risk
- Demonstrated network-driven risk amplification up to 400x
- Built a memory-efficient spare implementation scalable to real-world healthcare data

---

## Project Overview

This project applies the PageRank algorithm to a large-scale healthcare provider referral network to identify influential providers and analyze how risk propagates through referral relationships. Using the 2015 CMS Physician Shared Patient Patterns dataset, this analysis includes:
- 1,034,612 healthcare providers
- 65,739,582 referral relationships.
This project emphasizes scalability, numerical stability, and domain-specific interpretation of network centrality.

### From Coursework to Large-Scale Healthcare Analysis
This project extends a PageRank implementation originally developed for an upper-division applied linear algebra course (MAT 167).

Key extensions include:
- Scaling from ~280K web nodes to 1M+ healthcare providers
- Replacing dense matrices with SciPy sparse representations
- Implementing power iteration for large-scale convergence
- Adapting PageRank into a risk propagation framework
- Adding convergence benchmarking and performance analysis

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
- **Convergence speed:** 72 iterations for 1M+ node network
- **Memory efficiency:** Reduced from 8TB (dense) to 2GB (sparse), a 4000x improvement
- **Computation time:** ~39 mins from start to finish

---

## Motivation

Healthcare provider networks are complex ecosystems where referral patterns can reveal:
- **Quality improvement opportunities:** Identify key providers for targeted interventions
- **Fraud detection patterns:** Detect unusual referral behaviors
- **Care coordination optimization:** Understand information flow and patient handoffs
- **Risk propagation dynamics:** Model how issues cascade through provider relationships
Traditional healthcare analytics focus on individual providers. This project demonstrates that network position is equally critical for understanding provider influence and risk.

---

## Dataset

**Source:** [CMS Physician Shared Patient Patterns (2015)]([https://data.cms.gov/](https://www.nber.org/research/data/physician-shared-patient-patterns-data))
- **Providers:** 1,034,612 unique healthcare providers (NPIs)
- **Referral Relationships:** 65,739,582 connections
- **Network Density:** 6.14 × 10⁻⁵ (highly sparse)
- **Average Degree:** 63.54 referrals per provider

---

## Technical Implementation

### Algorithm: PageRank with Power Iteration

I implemented the PageRank algorithm using the **power iteration method**, which efficiently computes node importance in large networks.

**Core Algorithm:**
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
- **Challenge:** Dense matrix for 1M providers would require 1M² = 1 trillion entries (~8TB memory)
- **Solution:** Used `scipy.sparse.csc_matrix` to store only non-zero entries
- **Result:** Memory usage reduced from 8TB to ~2GB (4000x improvement)

#### 2. **Convergence Criterion**
- Used **L1 norm** instead of L2 norm for convergence checking
- Better suited for probability distributions
- Tolerance: `1e-8` provides excellent accuracy without excessive computation

#### 3. **Risk Propagation Model**
Adapted PageRank to model risk flow through the network:
- **Higher damping factor** (0.95 vs 0.85): Risk is "stickier" and stays in the network
- **Non-uniform initialization**: Starts with actual risk scores instead of uniform distribution
- **Composite risk metric**: Combines connectivity risk, referral imbalance, and isolation risk

### Risk Propagation Model

Adapted PageRank to model risk flow:
- **Higher damping factor (0.95)** to model persistent risk retention
- **Non-uniform initialization** using composite provider risk scores
- **Amplification factors** quantify how network position magnifies baseline risk
- **Top-percentile risk concentration analysis (90/95/99%)**

This approach identifies providers whose network position drives risk, even when their baseline risk is low.

---

## Methodology

### Analysis Pipeline

This project implements a full end-to-end healthcare network analysis system:

1. **Data Ingestion**
   - Loads CMS referral edge lists
   - Maps NPIs to matrix indices
   - Constructs memory-efficient sparse adjacency matrices

2. **Core PageRank Computation**
   - Power iteration with sparse linear algebra
   - Dangling-node handling
   - L1-norm convergence for probability stability

3. **Network Structure Analysis**
   - Hub (out-degree), Authority (in-degree), and PageRank centrality
   - Identification of top 10% providers by role
   - Network density and inequality metrics

4. **Risk Modeling & Propagation**
   - Composite risk score construction
   - PageRank-based risk diffusion model
   - Risk amplification analysis

5. **Performance & Convergence Testing**
   - Accuracy–runtime tradeoff analysis across tolerances
   - Iteration counts and timing benchmarks

6. **Visualization & Reporting**
   - Static plots and interactive dashboards
   - Centrality, risk, and convergence summaries

### Execution Details

#### Step 1: Data Loading & Graph Construction
```
Loading network data...
✓ File loaded successfully (26 seconds)
✓ Found 1,034,612 unique NPIs
✓ Mapped NPIs to matrix indices
✓ Constructed sparse adjacency matrix
```

**Technical Details:**
- Loaded edge list from CSV file
- Created bidirectional mapping: NPI ↔ matrix index
- Built sparse CSC matrix for efficient column operations

#### Step 2: PageRank Computation
```
Running PageRank algorithm...
✓ Converged in 72 iterations
✓ Computation time: 223.08 seconds (~3.7 minutes)
✓ Final error: 9.60 × 10⁻⁹
```

**Performance Characteristics:**
- **Iterations:** 72 (typical range: 50-100 for healthcare networks)
- **Convergence:** Achieved tolerance of 1e-8
- **Speed:** ~3.1 seconds per iteration for 1M+ node network

#### Step 3: Network Structure Analysis
Identified three categories of key providers:

| Category | Count | Description |
|----------|-------|-------------|
| **Hubs** | 102,610 | High outgoing referrals (top 10%) |
| **Authorities** | 102,812 | High incoming referrals (top 10%) |
| **Central** | 103,462 | High PageRank scores (top 10%) |

**Top Provider:** NPI 1003000126
- PageRank Score: High centrality
- Referrals Out: 213
- Referrals In: 218
- Role: Hub, Authority, and Central provider

#### Step 4: Risk Propagation Analysis
```
Running risk propagation model...
✓ Identified 103,462 high-risk providers (top 10%)
✓ Risk concentration: 54.21% in top 10%
✓ Mean amplification factor: Network position amplifies risk
```

**Risk Metrics:**
- **Mean Risk Score:** 9.67 × 10⁻⁷
- **Maximum Risk Score:** 0.001943
- **VaR 95%:** 2.89 × 10⁻⁶ (95% of providers below this risk level)
- **Expected Shortfall 95%:** 6.77 × 10⁻⁶ (average risk in worst 5%)

**Top 3 High-Risk Providers:**

| Rank | NPI | Risk Score | Amplification Factor |
|------|-----|------------|---------------------|
| 1 | 1538144910 | 0.001943 | **397.82x** |
| 2 | 1063497451 | 0.001307 | **398.57x** |
| 3 | 1700865094 | 0.001187 | **222.96x** |

*Amplification factor shows how much network position increases individual risk*

#### Step 5: Convergence Performance Testing
Tested algorithm across multiple tolerance levels to analyze accuracy-speed tradeoffs:

| Tolerance | Iterations | Time (s) | Converged |
|-----------|-----------|----------|-----------|
| 1e-04 | 22 | 71.22 | ✓ Yes |
| 1e-06 | 46 | 133.56 | ✓ Yes |
| **1e-08** | **72** | **212.05** | **✓ Yes** |
| 1e-10 | 99 | 299.00 | ✓ Yes |

**Key Insights:**
- **Diminishing returns:** Going from 1e-8 to 1e-10 increases iterations by 37% but only marginally improves accuracy
- **Optimal choice:** 1e-8 provides excellent accuracy with reasonable computation time
- **Scalability:** Linear relationship between tolerance and iterations

---

## Results & Visualizations

### 1. PageRank Distribution Analysis

**Finding:** Power-law distribution typical of real-world networks
- **Low-rank providers** (top 1-100) have significantly higher PageRank values
- **Distribution:** Highly concentrated, with most providers having low PageRank
- **Peak density:** Around PageRank value of 0 (most providers have minimal influence)

### 2. Risk Score Distribution

**Finding:** Extreme concentration of risk
- **Vast majority** of providers have near-zero risk scores
- **High-risk providers** (10%) account for **54.21%** of total network risk
- **Risk amplification:** Network position can amplify individual risk by up to **398x**

### 3. Risk Propagation Analysis

**Finding:** Network effects significantly amplify risk
- **Propagated risk** consistently higher than original risk scores
- **Network position matters:** Providers with many connections accumulate risk even if they start with low individual risk
- **High-risk count:** 103,462 providers (10%) identified as high-risk after propagation
- **Low-risk count:** 931,150 providers (90%) remain low-risk

**Interpretation:** This shows that network structure plays a crucial role in risk distribution. Providers who are well-connected or in central positions accumulate risk through their referral relationships.

---

## Installation & Usage

### Prerequisites
```bash
pip install numpy scipy matplotlib plotly networkx pandas
```

### Execution
```bash
python pagerank.py
```

### Expected Output
- Console output with analysis results
- Static visualizations in `visualizations/` folder
- Interactive HTML dashboard
- Execution log in `pagerank_analysis.log`

**Total Runtime:** ~39 minutes for full analysis on 1M+ provider network

---

## Project Structure

```
pagerank-healthcare/
│
├── pagerank2.py              # Core PageRank algorithm implementation
├── visualizations2.py        # Visualization generation module
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

## References
- Original Project Report: [Original PageRank Analysis (MAT 167)](original_pagerank/original_pagerank_report.pdf)
- Dataset: [CMS Physician Shared Patient Patterns (2015)]([https://data.cms.gov/](https://www.nber.org/research/data/physician-shared-patient-patterns-data))

---

## Author

**Angelina Cottone**
B.S. Statistics (Statistical Data Science), UC Davis 2025

---
*Last Updated: January 2026*
