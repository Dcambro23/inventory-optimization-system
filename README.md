# Inventory Optimization System for Walmart Retail Operations

> End-to-end demand forecasting and inventory optimization system combining time series analysis, machine learning, and mathematical optimization to minimize costs while maintaining service levels.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

---

## 🎯 Project Objective

Develop a **production-ready forecasting and optimization system** that predicts weekly retail demand and determines optimal inventory policies to minimize total costs (holding, stockout, ordering) while maintaining 95%+ service levels. 

This project demonstrates:
- ✅ End-to-end data science pipeline (EDA → Feature Engineering → Modeling → Optimization)
- ✅ Time series forecasting with multiple methodologies (Statistical, ML, Ensemble)
- ✅ Mathematical optimization with business constraints
- ✅ Financial impact quantification for stakeholder communication

**Target Audience:** Data Science roles at Microsoft, Google, and enterprise software companies

---

## 📊 Business Problem

Retailers face a critical trade-off:
- **Too much inventory** → Capital tied up, storage costs, obsolescence risk
- **Too little inventory** → Stockouts, lost sales, customer dissatisfaction

**Challenge:** Balance these competing objectives using data-driven forecasting and optimization.

**Solution Approach:**
1. Forecast demand with high accuracy (target: <10% MAPE on regular weeks)
2. Optimize inventory policies using forecasts + cost parameters
3. Quantify financial impact (% cost reduction, working capital freed)

---

## 📂 Dataset

**Source:** Walmart Store Sales (Kaggle)  
**Records:** 6,435 weekly observations  
**Stores:** 45 Walmart locations  
**Time Period:** February 2010 - October 2012 (143 weeks)  
**Features:** Sales, Temperature, Fuel Price, CPI, Unemployment, Holiday indicators

**Selected Stores for Deep Analysis:**
- **Store 20** (High Volume): $2.1M avg/week, stable pattern
- **Store 40** (Medium Volume): $964K avg/week, typical retail
- **Store 15** (Low Volume, Declining): $623K avg/week, -7% YoY trend

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.13 |
| **Data Processing** | pandas 2.3.3, numpy 2.3.3, scipy |
| **Visualization** | matplotlib 3.10, seaborn, plotly |
| **Statistical Models** | statsmodels 0.14.5 (ARIMA, Exponential Smoothing) |
| **Machine Learning** | scikit-learn 1.7.2, XGBoost 3.0.5, LightGBM |
| **Optimization** | PuLP 3.3.0 (Linear Programming) |
| **Development** | Jupyter, VS Code, Git/GitHub |

---

## 📁 Project Structure
```
inventory-optimization-system/
├── data/
│   ├── raw/                      # Original Walmart_Sales.csv (not tracked)
│   ├── processed/                # walmart_featured.csv, baseline_results.pkl
│   └── external/                 # Reference data (if needed)
├── notebooks/
│   ├── 01_data_exploration.ipynb          # ✅ EDA complete
│   ├── 02_feature_engineering.ipynb       # ✅ Feature engineering complete
│   ├── 03_baseline_models.ipynb           # ✅ Baseline models complete
│   ├── 04_advanced_forecasting.ipynb      # ⏳ In progress
│   └── ...                                # Future notebooks
├── src/                          # Reusable code modules (future)
│   ├── data/                     # Data loading functions
│   ├── models/                   # Model training/evaluation
│   └── visualization/            # Plotting utilities
├── reports/
│   └── figures/                  # High-quality visualizations
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip package manager
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/inventory-optimization-system.git
cd inventory-optimization-system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook
```

### Reproducing Results

1. Download the [Walmart Sales dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
2. Place `Walmart_Sales.csv` in `data/raw/`
3. Run notebooks sequentially: `01_data_exploration.ipynb` → `02_feature_engineering.ipynb` → `03_baseline_models.ipynb`

---

## 📈 Project Status

### ✅ Phase 1: Exploratory Data Analysis (Complete)
**Notebook:** `01_data_exploration.ipynb`

**Key Findings:**
- 📊 Massive seasonality: **December sales 39% higher than January**
- 🎄 Major holidays (Black Friday, Christmas): **40% sales lift**
- 📉 Minor holidays (Super Bowl, Labor Day): **~5% lift** (minimal impact)
- 🌡️ External variables (Temperature, Fuel Price, CPI, Unemployment): **Weak correlation <0.11**
- 🏪 Store heterogeneity: **8x range** between highest/lowest performers

**Strategic Decision:** Focus forecasting on temporal patterns and lag features; deprioritize external economic indicators.

---

### ✅ Phase 2: Feature Engineering (Complete)
**Notebook:** `02_feature_engineering.ipynb`

**Features Created (25+ total):**

| Category | Features | Purpose |
|----------|----------|---------|
| **Temporal** | Year, Month, Quarter, Week, DayOfYear | Calendar patterns |
| **Cyclical** | Month_sin/cos, Week_sin/cos | Capture seasonality without discontinuity |
| **Lag Features** | Sales_Lag_1, 2, 3, 4, 8, 12 | Autocorrelation (past predicts future) |
| **Rolling Stats** | RollMean_4/8/12, RollStd_4/8/12 | Trend smoothing + volatility |
| **Momentum** | WoW_Change, Momentum_4w, Diff_RollMean | Rate of change indicators |
| **Holiday Taxonomy** | Is_Major_Holiday, Is_Post_Holiday, Is_Pre_Holiday | Differentiate high-impact periods |

**Data Quality:**
- Original: 429 rows (3 stores × 143 weeks)
- After cleaning: **393 rows** (removed 8.4% with NaN from lag features)
- **Zero missing values** in final dataset

---

### ✅ Phase 3: Baseline Forecasting Models (Complete)
**Notebook:** `03_baseline_models.ipynb`

**Models Evaluated:**
1. Naive Forecast (Lag-1)
2. Moving Average (4-week)
3. Moving Average (12-week)
4. Simple Exponential Smoothing
5. Holt's Linear Trend
6. Holt-Winters (Triple Exponential Smoothing)

**Results:**

| Store | Best Model | MAPE | RMSE | MAE |
|-------|-----------|------|------|-----|
| **Store 20** (High Vol) | Simple Exp Smoothing | **3.72%** ✅ | $102,833 | $78,577 |
| **Store 40** (Medium) | Moving Avg (4w) | **6.19%** ✅ | $70,586 | $61,611 |
| **Store 15** (Declining) | Naive (Lag-1) | **4.71%** ✅ | $38,552 | $28,729 |
| **Overall Average** | Moving Avg (4w) | **5.17%** 🏆 | - | - |

**Key Insights:**
1. ✅ **Exceptional baseline performance** - MAPE <7% across all stores (better than expected)
2. 🎯 **Simple methods win** - Exponential Smoothing and Moving Averages outperform complex Holt's Linear (15% MAPE ❌)
3. 📉 **Holt-Winters failed** - Overfitting due to insufficient data for seasonal decomposition (MAPE 8-13%)
4. 🔄 **High autocorrelation confirmed** - Naive (Lag-1) competitive (4-7% MAPE), validates lag features strategy
5. 📊 **Store 15 paradox** - Declining trend captured implicitly by Lag-1; explicit trend modeling (Holt's) performed worse

**Benchmark Established:** Advanced models (ARIMA, XGBoost) must achieve **<5% MAPE** to justify added complexity.

---

### ⏳ Phase 4: Advanced Forecasting (In Progress)
**Target Notebook:** `04_advanced_forecasting.ipynb`

**Planned Models:**
- SARIMA (Seasonal ARIMA)
- Prophet (Facebook's forecasting tool)
- XGBoost Regression (with engineered features)
- LightGBM (comparison)

**Performance Targets:**

| Store | Baseline MAPE | Target ARIMA | Target XGBoost |
|-------|---------------|--------------|----------------|
| Store 20 | 3.72% | **<3.0%** | **<2.5%** |
| Store 40 | 6.19% | **<5.0%** | **<4.0%** |
| Store 15 | 4.71% | **<4.0%** | **<3.0%** |

---

### 🔮 Phase 5: Inventory Optimization (Planned)
**Target Notebook:** `06_inventory_optimization.ipynb`

**Approach:**
- Use best forecasting model to generate demand predictions
- Formulate inventory optimization as Linear Programming problem
- Objective: Minimize (Holding Cost + Stockout Cost + Ordering Cost)
- Constraints: Service level ≥95%, capacity limits

**Expected Outputs:**
- Optimal reorder points per store
- Optimal order quantities
- Safety stock requirements (especially for major holidays)
- Cost reduction quantification vs naive policies

---

### 🎯 Phase 6: Business Impact Quantification (Planned)
**Target Notebook:** `08_business_impact.ipynb`

**Deliverables:**
- Financial impact assessment (% cost reduction)
- Working capital freed up (dollar amount)
- Sensitivity analysis on cost parameters
- Executive summary with recommendations

**Target Claim:**
> "Optimized inventory policies reduce costs by **15-20%** while maintaining **95% service level**, freeing up **$X million** in working capital across 45 stores."

---

## 🎓 Key Learnings & Methodology

### Why This Project Structure?

**1. Baseline-First Approach**
- Establish simple benchmarks before complex models
- Demonstrates rigor (not jumping to "fancy" algorithms)
- Provides interpretable fallback if advanced models fail

**2. Evidence-Driven Decisions**
- External variables excluded based on **correlation analysis**, not assumptions
- Holiday taxonomy created after **observing 40% vs 5% lift** in EDA
- Store selection via **stratified sampling** (not arbitrary)

**3. Production Thinking**
- Train/test splits respect **temporal ordering** (no leakage)
- Reproducible pipeline (requirements.txt, clear notebook sequence)
- Modular design (future refactoring into src/ modules)

---

## 📊 Visualizations

### Sample Outputs (Notebook 03)

**Store 20: Baseline Models Comparison**
- Simple Exponential Smoothing tracks actuals closely (3.72% MAPE)
- Holt-Winters shows erratic predictions (overfitting)
- Major holiday spikes captured reasonably well

**Residual Analysis**
- Mean residual near zero (unbiased forecasts)
- Larger errors during holiday transitions (expected)
- Store 15 residuals more stable than anticipated

*(Full visualizations available in `reports/figures/` after running notebooks)*

---

## 🔧 Technical Decisions Log

### Major Design Choices

**1. Store Selection: 3 vs 45**
- **Decision:** Deep analysis of 3 representative stores
- **Rationale:** Depth > breadth for portfolio; demonstrates sampling strategy
- **Selection:** High/Medium/Low volume via stratified sampling

**2. External Variables: Exclude from modeling**
- **Decision:** Deprioritize Temperature, Fuel Price, CPI, Unemployment
- **Evidence:** All correlations <0.11 with Weekly_Sales
- **Benefit:** Simpler models, faster training, easier interpretation

**3. Lag Features: 1, 2, 3, 4, 8, 12 weeks**
- **Decision:** Not yearly lag (52) to avoid losing first year of data
- **Rationale:** Capture short-term (1-4), bi-monthly (8), quarterly (12) patterns
- **Cost:** 8.4% data loss from NaN, acceptable for pattern capture

**4. Missing Values: Drop vs Impute**
- **Decision:** Drop 36 rows with NaN in lag features
- **Alternatives Rejected:** Forward fill (artificial correlation), mean imputation (inappropriate for time series)
- **Justification:** Preserves data integrity, 8.4% loss acceptable

---

## 🚀 Future Enhancements

**If Time Permits (before applications):**
- [ ] Implement LSTM/Transformer models for comparison
- [ ] Multi-step forecasting (1-week, 4-week, 12-week horizons)
- [ ] Store clustering (group similar stores for shared models)
- [ ] Interactive dashboard (Streamlit/Plotly Dash)
- [ ] Blog post explaining methodology (Medium/LinkedIn)

**Production Considerations (not in scope):**
- API layer for serving predictions
- Automated retraining pipeline (Airflow/Prefect)
- Model monitoring and drift detection
- A/B testing framework for policy changes

---

## 👤 Author

**Diego** - Industrial Engineer specializing in Supply Chain Analytics and Data Science

**Background:**
- 8 years experience in logistics operations (family business)
- Pursuing Licentiate in Industrial Engineering (Logistics & Supply Chain focus)
- Transitioning to Data Science roles at Big Tech (target: Microsoft)

**Connect:**
- GitHub: Dcambro23 (https://github.com/Dcambro23)
- LinkedIn: Diego Alonso Cambronero Zeledón (https://linkedin.com/in/diegocambronero)

---

## 📝 License

This project is licensed under the MIT License - see the MIT (LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Walmart Store Sales - Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
- Inspiration: Real-world retail analytics challenges
- Tools: Open-source Python data science ecosystem

---

## 📚 References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.)
- Statsmodels Documentation: Time Series Analysis
- Scikit-learn User Guide: Model Evaluation
- PuLP Documentation: Optimization Modeling

---

**Last Updated:** October 19, 2025  
**Project Timeline:** Week 3 of 16 (estimated completion: March 2026)