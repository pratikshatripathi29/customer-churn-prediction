
📉 Customer Churn Prediction (Telco) – End-to-End Machine Learning Project

Predict which telecom customers are likely to leave, explain why, and choose a decision threshold that maximises retention profit.
<img width="1861" height="981" alt="image" src="https://github.com/user-attachments/assets/3a1c860c-a55f-44a7-9985-b39bab3d14b0" />
<img width="1618" height="910" alt="image" src="https://github.com/user-attachments/assets/1df51b1d-d119-450e-8ece-a5f81a58520d" />
<img width="1677" height="928" alt="image" src="https://github.com/user-attachments/assets/c3ede284-0c46-4e4e-86c0-0a4a9d679a6a" />
<img width="1617" height="917" alt="image" src="https://github.com/user-attachments/assets/6b4454cf-16eb-4953-840a-2210ec10b0db" />
🎯 Business Problem

Acquiring a new customer costs far more than retaining an existing one. The goal is to identify high-risk customers early so a retention team can target them with offers instead of discounting everyone.

📊 Dataset
Source: IBM Telco Customer Churn (Kaggle)
Size: 7,043 customers, 21 columns (demographics, services, contract, billing)
Target: Churn (Yes/No), about 26.5% churners (imbalanced)
🔬 Approach
Data cleaning: fixed TotalCharges stored as text (blank for new customers), dropped the ID column, encoded the target
EDA: class balance, numeric distributions, churn rate per category, correlations
Feature engineering: tenure_group, num_services, avg_monthly_spend, charge_gap
Leakage-safe preprocessing: ColumnTransformer (StandardScaler + OneHotEncoder) inside a scikit-learn Pipeline
Model comparison: Logistic Regression, Random Forest, XGBoost with 5-fold stratified CV
Imbalance handling: class_weight='balanced' / scale_pos_weight
Tuning: RandomizedSearchCV (30 iterations) optimising ROC-AUC
Evaluation: ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix on a held-out stratified test set
Business threshold tuning: picked the cut-off maximising expected profit (retention offer cost vs customer value)
Explainability: permutation importance and SHAP
Deployment prep: risk tiers (Low/Medium/High) and the model saved with joblib
✅ Results
Model (5-fold CV, training data)	ROC-AUC
Logistic Regression	0.846
XGBoost	0.842
Random Forest	0.829

Tuned XGBoost on the held-out test set: ROC-AUC 0.847, PR-AUC 0.667, recall about 80% on churners at the default 0.5 threshold.

A simple Logistic Regression performs almost as well as tuned XGBoost, which shows the signal in this dataset is largely linear and interpretable.

⚠️ The business-threshold step uses assumed costs (offer cost $20, saved-customer value $200, 40% offer success). Replace them with real company figures.

💡 Key Insights
Month-to-month contracts churn far more than 1- or 2-year contracts.
New customers (low tenure) are the highest-risk group, especially in the first year.
Fiber-optic internet and electronic-check payments are linked to higher churn.
Customers without tech support / online security churn more; more bundled services means stickier customers.

Recommendation: target new month-to-month customers with contract-upgrade offers and bundle tech support / security.

🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn (Pipelines, ColumnTransformer, RandomizedSearchCV) · XGBoost · SHAP · Joblib · Jupyter Notebook · VS Code · Git/GitHub

📁 Project Structure
├── churn_prediction.ipynb     # Full analysis and modelling notebook
├── requirements.txt
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv   # auto-downloaded on first run
└── models/
    └── churn_model.pkl            # saved model + optimal threshold
▶️ How to Run
bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb   # or open the .ipynb in VS Code and Run All

The dataset downloads automatically on the first run.

🚀 Future Improvements
Deploy with Streamlit for live churn scoring
Compare SMOTE vs class weighting; try LightGBM / CatBoost
Probability calibration and cost-sensitive learning
Monitor data drift and retrain periodically
