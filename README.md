KNN-Diabetes-Prediction-System


> **An end-to-end clinical machine learning pipeline** that predicts diabetes risk from patient biometrics — featuring expert feature engineering, outlier-robust preprocessing, and an optimized KNN classifier.



##  Table of Contents

- [ Project Overview](#-project-overview)
- [ Key Highlights](#-key-highlights)
- [ Pipeline Architecture](#️-pipeline-architecture)
- [ Exploratory Data Analysis](#-exploratory-data-analysis)
- [ Feature Engineering](#-feature-engineering)
- [ Model & Evaluation](#-model--evaluation)
- [ Project Structure](#️-project-structure)
- [ Getting Started](#-getting-started)
- [ Dependencies](#-dependencies)
- [ Real-World Applications](#-real-world-applications)
- [ Roadmap](#️-roadmap)
- [ Contributing](#-contributing)


---

##  Project Overview

Diabetes affects over **537 million adults worldwide** and is one of the leading causes of preventable deaths. Early prediction is key to intervention.

This project builds a **full ML classification pipeline** on the **Pima Indians Diabetes Dataset** using the **K-Nearest Neighbors (KNN)** algorithm. It goes beyond a basic model — incorporating domain-informed feature engineering, robust preprocessing, and systematic hyperparameter tuning to maximize predictive performance.

```
Patient Biometrics → Preprocessing → Feature Engineering → KNN Model → Diabetes Risk Prediction
     (Raw Data)         (Clean)          (10 Features)       (Tuned)       (0: No / 1: Yes)
```

---

##  Key Highlights

|  What Makes This Project Stand Out |
|---|
|  **Domain-driven feature engineering** — custom interaction terms (Glucose×BMI, Risk Score, Insulin Resistance) built from medical knowledge |
|  **Robust preprocessing** — median imputation for biologically invalid zeros + IQR-based outlier removal |
|  **Hyperparameter tuning** — empirical K sweep (1–24) with accuracy curve visualization |
|  **Distance-weighted KNN** — closer neighbors weighted higher using Manhattan distance |
|  **Production-ready structure** — clean, modular, and reproducible notebook pipeline |
|  **Rich EDA** — correlation heatmaps, boxplots, pairplots, and histograms |

---

##  Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                            │
│              Pima Indians Diabetes Dataset (CSV)                 │
│               768 samples · 8 original features                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXPLORATORY DATA ANALYSIS                     │
│    Histograms · Correlation Heatmap · Boxplots · Pairplot       │
│    Class Distribution · Feature-Outcome Relationships           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                             │
│   ① Zero Imputation  →  Median replacement (5 columns)         │
│   ② IQR Outlier Removal  →  Glucose, BMI, Insulin              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                          │
│    Glucose_BMI · Insulin_Glucose_Ratio · Risk_Score            │
│    Pregnancy_Age · Insulin_Log · DPF_Log                       │
│                  10 final features selected                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  TRAIN / TEST SPLIT  (80 / 20)                  │
│                   StandardScaler Normalization                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              KNN CLASSIFIER (n=15, Manhattan, Distance)         │
│                 K Tuning Curve · Accuracy Score                 │
│                    Classification Report                         │
└──────────────────────────────────────────────────────────────────┘
```



##  Exploratory Data Analysis

The EDA phase covers a comprehensive visual and statistical analysis:

| Visualization | Purpose |
|---|---|
|  **Feature Histograms** | Distribution shape and skewness of all 8 features |
|  **Correlation Heatmap** | Identify multicollinearity and key predictors |
|  **Boxplots (per class)** | Glucose, BMI, and Age distributions by Outcome |
|  **Pairplot** | Pairwise relationships coloured by diabetes status |
|  **Scatter Plot** | Glucose vs BMI split by Outcome |
|  **Null Audit** | Missing value check across all columns |

> **Key Insight:** `Glucose`, `BMI`, and `Age` show the strongest separation between diabetic and non-diabetic patients.



##  Feature Engineering

Six new features were crafted using **medical domain knowledge** to boost signal:

```python
# Interaction between the two strongest predictors
df['Glucose_BMI'] = df['Glucose'] * df['BMI']

# Insulin resistance proxy
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)

# Composite clinical risk flag (0, 1, 2, or 3)
df['Risk_Score'] = (
    (df['Glucose'] > 140).astype(int) +   # Hyperglycemia threshold
    (df['BMI'] > 30).astype(int)      +   # Obese classification
    (df['Age'] > 40).astype(int)          # Age-related risk
)

# Reproductive history × age interaction
df['Pregnancy_Age'] = df['Pregnancies'] * df['Age']

# Log transforms to normalize right-skewed distributions
df['Insulin_Log'] = np.log1p(df['Insulin'])
df['DPF_Log']     = np.log1p(df['DiabetesPedigreeFunction'])
```

**Final Feature Set (10 features):**

| # | Feature | Type | Clinical Meaning |
|---|---|---|---|
| 1 | `Glucose` | Original | Plasma glucose concentration |
| 2 | `BMI` | Original | Body mass index |
| 3 | `Age` | Original | Patient age |
| 4 | `Pregnancies` | Original | Number of pregnancies |
| 5 | `Glucose_BMI` | Engineered | Multiplicative glucose-obesity risk |
| 6 | `Risk_Score` | Engineered | Composite clinical risk (0–3) |
| 7 | `Insulin_Glucose_Ratio` | Engineered | Insulin resistance proxy |
| 8 | `Pregnancy_Age` | Engineered | Reproductive history × age |
| 9 | `Insulin_Log` | Engineered | Log-normalized insulin level |
| 10 | `DPF_Log` | Engineered | Log-normalized diabetes pedigree |

---

##  Model & Evaluation

### Model Configuration

```python
KNeighborsClassifier(
    n_neighbors = 15,          # Optimal K selected from tuning curve (K = 1–24)
    metric      = 'manhattan', # L1 distance — robust to outliers
    weights     = 'distance'   # Closer neighbours carry more weight
)
```

### Why KNN?

-  **Interpretable** — predictions are explainable by nearest training examples
-  **Non-parametric** — no assumptions about data distribution
-  **Distance-weighted** — reduces noise influence from farther neighbors

### Preprocessing Steps

| Step | Method | Detail |
|---|---|---|
| Zero Imputation | Median replacement | Applied to 5 clinically invalid zero columns |
| Outlier Removal | IQR filtering | 1.5×IQR bounds on Glucose, BMI, Insulin |
| Feature Scaling | StandardScaler | Fit on train only — no data leakage |
| Train/Test Split | 80 / 20 | `random_state=42` for full reproducibility |

### Evaluation Output

```
Accuracy Score        →  Reported on held-out test set (20%)
Classification Report →  Precision · Recall · F1-Score per class
K Tuning Curve        →  Accuracy plotted for K = 1 to 24
```

---

##  Project Structure

```
KNN-Diabetes-Prediction-System/
│
├──  KNN-Diabetes-Prediction-System.ipynb   ← Complete ML pipeline notebook
│
├──  data/
│   └── diabetes.csv                          ← Pima Indians Diabetes Dataset
│
├──  requirements.txt                       ← Python dependencies
└──  README.md
```

---

##  Getting Started

###  Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/KNN-Diabetes-Prediction-System.git
cd KNN-Diabetes-Prediction-System
```

### 2️ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️ Launch the Notebook

```bash
jupyter notebook KNN-Diabetes-Prediction-System.ipynb
```

> ⚡ Update the CSV path in the data loading cell to your local `diabetes.csv`, then run **Kernel → Restart & Run All**.

---

##  Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

##  Real-World Applications


** Clinical Decision Support**
Flag high-risk patients for early glucose testing in primary care — reducing time-to-diagnosis and enabling preventive care at scale.



** Population Health Screening**
Batch-score anonymized patient records to identify at-risk cohorts for targeted public health intervention programs.



** Health App Integration**
Export the trained model via Joblib and embed it into a mobile or web health app where users receive a personalised diabetes risk assessment.



** ML Education & Reference**
A clean, well-documented end-to-end pipeline for learning supervised classification, feature engineering, and clinical ML in Python.



##  Roadmap

- [ ]  Cross-validation (StratifiedKFold) for robust accuracy estimation
- [ ]  SMOTE oversampling to handle class imbalance
- [ ]  Model benchmarking — Logistic Regression, Random Forest, XGBoost
- [ ]  SHAP explainability for feature importance visualization
- [ ]  Streamlit web app for interactive patient risk prediction
- [ ]  FastAPI REST endpoint (`/predict`) for production deployment
- [ ]  Docker containerization for portable, reproducible deployment



##  Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


##  Acknowledgements

- [UCI / Kaggle — Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Scikit-learn](https://scikit-learn.org/) — Machine learning in Python
- [Seaborn](https://seaborn.pydata.org/) — Statistical data visualization
- [Pandas](https://pandas.pydata.org/) — Data manipulation and analysis



Author
Sarfaraz Ali



</div>
