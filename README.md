# Healthcare Network PageRank Analysis

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Sparse-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Project Overview

This project applies the PageRank algorithm to analyze a massive healthcare provider network, identifying influential providers and modeling how risk propagates through referral relationships. Using real-world data from the 2015 Physician Shared Patient Patterns dataset, I analyzed over 1 million healthcare providers and 65 million referral connections.

### Key Objectives
- Identify influential healthcare providers using network centrality measures
- Model risk propagation through provider referral networks
- Analyze network structure to categorize providers by role (hubs, authorities, central nodes)
- Optimize algorithm performance for large-scale network analysis

---

## Motivation

Healthcare networks are complex systems where providers refer patients to specialists and other care facilities. Understanding these networks can:
- **Identify key providers** for quality improvement interventions
- **Detect unusual referral patterns** that may indicate fraud or inefficiency
- **Optimize care coordination** by understanding network structure
- **Model risk propagation** to understand how issues spread through the network

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

---

## Dataset

**Source:** [CMS Physician Shared Patient Patterns (2015)]([https://data.cms.gov/](https://www.nber.org/research/data/physician-shared-patient-patterns-data))

**Specifications:**
- **Providers:** 1,034,612 unique healthcare providers (NPIs)
- **Referral Relationships:** 65,739,582 connections
- **Network Density:** 6.14 × 10⁻⁵ (highly sparse)
- **Average Degree:** 63.54 referrals per provider
- **Data Processing Time:** ~4 minutes to load and construct sparse matrix

---

## Methodology & Process

### Step 1: Data Loading & Graph Construction
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

### Step 2: PageRank Computation
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

### Step 3: Network Structure Analysis
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

### Step 4: Risk Propagation Analysis
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

### Step 5: Convergence Performance Testing
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

## Key Insights & Business Value

### 1. Network Centralization
- **Finding:** Network is highly centralized with a small number of influential providers
- **Implication:** Interventions targeting top 10% of providers could impact majority of network

### 2. Risk Concentration
- **Finding:** 54% of risk concentrated in 10% of providers
- **Implication:** Risk management efforts should focus on high-centrality providers

### 3. Amplification Effects
- **Finding:** Network position can amplify individual risk by up to 400x
- **Implication:** Provider risk assessment should consider network context, not just individual factors

### 4. Algorithm Efficiency
- **Finding:** PageRank converges quickly (72 iterations) even for 1M+ node networks
- **Implication:** This approach scales to real-world healthcare networks

---

## Technologies Used

- **Python 3.7+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **SciPy (sparse)**: Sparse matrix operations for memory efficiency
- **Matplotlib**: Static publication-quality visualizations
- **Plotly**: Interactive HTML dashboards
- **NetworkX**: Network topology visualization (optional)
- **Pandas**: Data manipulation and correlation analysis

---

## Project Structure

```
pagerank-healthcare/
│
├── pagerank2.py              # Core PageRank algorithm implementation
├── visualizations2.py        # Visualization generation module
├── physician-shared-patient-patterns-2015-days180.txt  # Dataset
├── visualizations/           # Generated plots and dashboards
│   ├── healthcare_pagerank_analysis.png
│   ├── healthcare_risk_analysis.png
│   ├── convergence_analysis.png
│   └── healthcare_network_dashboard.html
├── pagerank_analysis.log     # Execution logs
└── README.md                 # This file
```

---

## Skills Demonstrated

### Technical Skills
- **Graph Algorithms:** PageRank, centrality measures, network analysis
- **Optimization:** Sparse matrix operations, memory-efficient data structures
- **Numerical Methods:** Power iteration, convergence analysis, error metrics
- **Data Processing:** Large-scale data loading, indexing, transformation
- **Visualization:** Multi-panel plots, interactive dashboards, professional presentation

### Analytical Skills
- **Problem Decomposition:** Breaking complex network analysis into modular components
- **Performance Analysis:** Benchmarking, tradeoff analysis, optimization
- **Risk Modeling:** Adapting algorithms for domain-specific applications
- **Statistical Analysis:** Distribution analysis, percentile calculations, correlation

### Software Engineering
- **Code Quality:** Comprehensive documentation, logging, error handling
- **Modularity:** Reusable functions, clear separation of concerns
- **Testing:** Convergence validation, edge case handling
- **Version Control:** Git workflow, professional repository structure

---

## 🚀 Running the Project

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

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Network Size** | 1,034,612 providers |
| **Edge Count** | 65,739,582 referrals |
| **PageRank Iterations** | 72 |
| **Convergence Time** | 223.08 seconds |
| **Total Runtime** | 2,333.83 seconds (~39 min) |
| **Memory Usage** | ~2GB (sparse matrices) |
| **Convergence Error** | 9.60 × 10⁻⁹ |

---

## Future Enhancements

- [ ] **Temporal Analysis:** Track network evolution over multiple years
- [ ] **Community Detection:** Identify provider clusters and specialization groups
- [ ] **Predictive Modeling:** Use network features to predict provider outcomes
- [ ] **Real-time Updates:** Incremental PageRank for dynamic networks
- [ ] **Geographic Analysis:** Incorporate provider location data
- [ ] **Parallel Processing:** Distribute computation across multiple cores

---

## References

- CMS Physician Shared Patient Patterns Dataset (2015)

---

## Author

**Angelina Cottone**

---

## Acknowledgments

- Centers for Medicare & Medicaid Services (CMS) for providing the dataset

---

*Last Updated: January 2026*


