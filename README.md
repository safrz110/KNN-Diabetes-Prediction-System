KNN Diabetes Prediction System
Python KNN Scikit-learn Pandas Seaborn License
A machine learning pipeline for predicting diabetes in patients using the K-Nearest Neighbors (KNN) algorithm with feature engineering, outlier handling, and hyperparameter tuning.
Live Demo · Documentation · Report Bug · Request Feature

Table of Contents

Overview
Features
Tech Stack
Project Structure
Installation
Usage
Model Configuration
Dataset Features
Applications
Contributing


Overview
This project implements a diabetes prediction system using the K-Nearest Neighbors (KNN) classification algorithm on the Pima Indians Diabetes Dataset. The pipeline covers:

Loading and exploring the dataset with summary statistics and visualizations
Handling missing/zero values and removing outliers using IQR filtering
Engineering domain-specific interaction features to boost model performance
Scaling features with StandardScaler and tuning K via accuracy curves
Evaluating the final model with accuracy score and classification report

All computation runs on standard CPU hardware — no GPU required.

Features
FeatureDetailsKNN ClassificationConfigurable K, distance metric, and weighting schemeFeature EngineeringGlucose×BMI, Insulin/Glucose ratio, Risk Score, log transformsOutlier RemovalIQR-based filtering on Glucose, BMI, and InsulinZero ImputationReplaces biologically invalid zeros with column mediansK Tuning CurveAccuracy vs K plot for K = 1–24Correlation HeatmapSeaborn heatmap of all feature correlationsPairplot AnalysisPairwise scatter plots colored by OutcomeClassification ReportPrecision, recall, F1-score per class

Tech Stack
┌─────────────────────────────────────────────────────────┐
│                  Machine Learning                       │
│             Scikit-learn (KNN, Pipeline)                │
├──────────────────────┬──────────────────────────────────┤
│   Data Processing    │      Visualization               │
│   Pandas · NumPy     │    Matplotlib · Seaborn          │
├──────────────────────┴──────────────────────────────────┤
│                  Model Persistence                      │
│                      Joblib                             │
└─────────────────────────────────────────────────────────┘

Project Structure
KNN-Diabetes-Prediction-System/
│
├── KNN-Diabetes-Prediction-System.ipynb   # Main Jupyter notebook
│
├── diabetes.csv                           # Pima Indians Diabetes Dataset
│
├── requirements.txt                       # Python dependencies
└── README.md

Installation
Prerequisites

Python 3.10 or higher
pip / conda
Jupyter Notebook or JupyterLab

Step 1 — Clone the Repository
bashgit clone https://github.com/YOUR_USERNAME/KNN-Diabetes-Prediction-System.git
cd KNN-Diabetes-Prediction-System
Step 2 — Create a Virtual Environment
bash# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Or using conda
conda create -n knn-diabetes python=3.10
conda activate knn-diabetes
Step 3 — Install Dependencies
bashpip install -r requirements.txt
Step 4 — Launch the Notebook
bashjupyter notebook KNN-Diabetes-Prediction-System.ipynb

Usage
Running the Full Pipeline

Open the notebook in Jupyter
Update the dataset path in the data loading cell to point to your local diabetes.csv
Run all cells from top to bottom (Kernel → Restart & Run All)
Review the EDA plots, accuracy output, and classification report

Notebook Sections
SectionDescriptionData LoadingRead CSV, inspect shape, info, and descriptive statsEDAHistograms, correlation heatmap, boxplots, pairplotPreprocessingZero imputation, IQR outlier removalFeature EngineeringInteraction terms, risk score, log transformsTrain / Test Split80/20 split with random_state=42ScalingStandardScaler fit on train, transform on testModel TrainingKNN with n_neighbors=15, metric='manhattan', weights='distance'EvaluationAccuracy score, classification reportK TuningAccuracy vs K curve (K = 1–24)

Model Configuration
ParameterValueDescriptionAlgorithmKNeighborsClassifierK-Nearest Neighborsn_neighbors15Number of neighborsmetricmanhattanDistance metricweightsdistanceCloser neighbors weighted moreTest size20%Train/test split ratioScalerStandardScalerZero mean, unit variance

Dataset Features
The model is trained on the following engineered feature set derived from the Pima Indians Diabetes Dataset:
FeatureTypeDescriptionGlucoseOriginalPlasma glucose concentrationBMIOriginalBody mass indexAgeOriginalPatient age in yearsPregnanciesOriginalNumber of pregnanciesGlucose_BMIEngineeredGlucose × BMI interactionRisk_ScoreEngineeredSum of high-glucose + high-BMI + age>40 flagsInsulin_Glucose_RatioEngineeredInsulin / (Glucose + 1)Pregnancy_AgeEngineeredPregnancies × AgeInsulin_LogEngineeredlog1p(Insulin)DPF_LogEngineeredlog1p(DiabetesPedigreeFunction)

Zero Imputation: Biologically invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are replaced with their column medians before modeling.


Applications
🏥 Clinical Decision Support
Assist healthcare providers in flagging high-risk patients for further glucose testing, reducing time to diagnosis in primary care settings.
📊 Population Health Screening
Deploy as a batch scoring tool on anonymized patient records to identify at-risk cohorts for preventive intervention programs.
🔬 Medical Research
Use the engineered features (Risk Score, Glucose×BMI) as a baseline feature set for more advanced models such as Random Forest or XGBoost in diabetes research.
🎓 Educational Resource
A clean, well-commented KNN pipeline for students learning supervised classification, feature engineering, and model evaluation in Python.
📱 Health App Integration
Embed the trained model (exported via Joblib) into a mobile or web health application where users input their biometrics to receive a risk assessment.

Roadmap

 Hyperparameter tuning with GridSearchCV / cross-validation
 Comparison with Logistic Regression, Random Forest, and SVM
 SMOTE oversampling for class imbalance
 Streamlit web app for interactive predictions
 Model export and deployment via FastAPI
 SHAP values for feature importance and explainability


Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request


License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgements

Pima Indians Diabetes Dataset — UCI Machine Learning Repository via Kaggle
Scikit-learn — Machine learning in Python
Seaborn — Statistical data visualization
Pandas — Data analysis and manipulation


Built with ❤️ using KNN + Scikit-learn
Author: Sarfaraz Ali
