import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

df=pd.read_csv(r"C:\Users\Kirti\OneDrive\Desktop\Loan Default Prediction\data\processed\cleaned_loan_data.csv")

#Feature Engineering
df['loan_income_ratio']=df['loan_amount']/(df['annual_income']+1)
df['total_delinquency_impact'] = df['delinquency_history'] + df['num_of_delinquencies']
df.drop(columns=['delinquency_history'], inplace=True)

#Converting grade_subgrade
def convert_grade(value):
    grade=value[0]
    sub=int(value[1])

    grade_map={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6}
    return grade_map[grade]*10+sub

df['grade_subgrade']=df['grade_subgrade'].apply(convert_grade)

#Ctaegorical encoding
cat_cols = ['employment_status', 'loan_purpose']
#Education encoding
edu_order=[["high school","bachelor's","master's","phd","other"]]


#Spliting columns requiring scaling an dnon scaling
scale_cols = [
    'annual_income', 'loan_amount', 'installment', 
    'total_credit_limit', 'current_balance', 
    'loan_income_ratio', 'debt_to_income_ratio'
]

non_scale_cols = [
    'age', 'credit_score', 'interest_rate', 'loan_term', 
    'grade_subgrade', 'num_of_open_accounts', 'public_records', 
    'num_of_delinquencies', 'total_delinquency_impact'
]

#Feature & target
x=df.drop("loan_paid_back",axis=1)
y=df["loan_paid_back"]


#Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ('edu', OrdinalEncoder(categories=edu_order), ['education_level']),
    ('scale_num', RobustScaler(), scale_cols),
    ('pass_num', 'passthrough', non_scale_cols)
])

#Final model

final_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    scale_pos_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_lambda=2,
    early_stopping_rounds=50,
    random_state=42,
    eval_metric='logloss'
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', final_model)
])

#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

preprocessor.fit(x_train, y_train)
x_test_transformed = preprocessor.transform(x_test)

pipeline.fit(
    x_train, y_train,
    model__eval_set=[(preprocessor.transform(x_test), y_test)],
    model__verbose=False
)

y_prob_xgb = pipeline.predict_proba(x_test)[:, 1]
threshold = 0.94
y_pred_xgb = (y_prob_xgb >= threshold).astype(int)


train_acc = pipeline.score(x_train, y_train) * 100
test_acc = pipeline.score(x_test, y_test) * 100
print(f"Standard Train Accuracy (0.5 threshold): {train_acc:.2f}%")
print(f"Standard Test Accuracy (0.5 threshold): {test_acc:.2f}%")

with open(r"C:\Users\Kirti\OneDrive\Desktop\Loan Default Prediction\models\full_pipeline.pkl","wb") as f:
    pickle.dump(pipeline,f)

print("Pipeline trained and saved successfully.")