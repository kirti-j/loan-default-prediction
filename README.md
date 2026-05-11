🏦 Loan Default Prediction System

A professional Machine Learning project that predicts whether a borrower will default on a loan using financial, demographic, and credit-related features. This project demonstrates a complete ML pipeline from data preprocessing to model optimization.


📌 Project Overview

Loan default prediction is a critical problem in the finance industry. This system helps identify high-risk borrowers, enabling better decision-making for loan approvals and risk management.


🎯 Objective

To build a machine learning model that accurately predicts loan repayment behavior (Default vs Non-Default) with strong performance on unseen data.

⚙️ Project Workflow

1️⃣ Data Preprocessing
- Removed duplicates and standardized column names  
- Cleaned categorical data  
- Verified dataset with no missing values  

2️⃣ Feature Engineering
- Created `loan_income_ratio` feature  
- Applied:
  - Ordinal Encoding (education level)  
  - Custom encoding (loan grade/subgrade)  
  - One-Hot Encoding (categorical features)  
- Used **RobustScaler** to handle outliers  


3️⃣ Model Building
- Logistic Regression (Baseline)  
- Random Forest  
- XGBoost (Final Model)  

4️⃣ Hyperparameter Tuning
- Used RandomizedSearchCV to optimize XGBoost  
- Improved model performance and generalization  

🏆 Final Model Performance

| Metric   | Value   |
|----------|--------|
| Accuracy | 90.35%|

 📊 Key Insights

- Feature engineering improved prediction performance  
- Proper tuning reduced overfitting  
- Credit-related features are strong predictors  

🛠️ Tech Stack

- Language:Python  
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost  
- Visualization: Matplotlib, Seaborn  
- Tools: Jupyter Notebook  

🚀 How to Run

bash
git clone <your-repo-link>
cd loan-default-prediction
pip install -r requirements.txt

📢 Conclusion
This project demonstrates an end-to-end machine learning workflow for a real-world financial problem, achieving 90.35% accuracy with a well-generalized model.

👤 Author
Kirti Subhash Jadhav
SYBCA Student | Aspiring Data Analyst

⭐ If you found this project useful, consider giving it a star!
