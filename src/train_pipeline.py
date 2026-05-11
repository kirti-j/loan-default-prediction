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

#Converting grade_subgrade
def convert_grade(value):
    grade=value[0]
    sub=int(value[1])

    grade_map={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6}
    return grade_map[grade]*10+sub

df['grade_subgrade']=df['grade_subgrade'].apply(convert_grade)


#Education encoding
edu_order=[["high school","bachelor's","master's","phd","other"]]

#Feature & target
x=df.drop("loan_paid_back",axis=1)
y=df["loan_paid_back"]

#Categorical Column
cat_cols=['gender','marital_status','employment_status','loan_purpose']

num_cols=['age','annual_income','monthly_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate','loan_term','installment','grade_subgrade','num_of_open_accounts','total_credit_limit','current_balance','delinquency_history','public_records','num_of_delinquencies','loan_income_ratio']


#Preprocessing
preprocessor=ColumnTransformer([('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat_cols),
('edu',OrdinalEncoder(categories=edu_order),['education_level']),
('num',RobustScaler(),num_cols)
 ])

#Final model

model=XGBClassifier(
    subsample=0.8,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='logloss'
)

pipeline=Pipeline([('preprocessing',preprocessor),
('model',model)])

#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

pipeline.fit(x_train,y_train)

with open(r"C:\Users\Kirti\OneDrive\Desktop\Loan Default Prediction\models\full_pipeline.pkl","wb") as f:
    pickle.dump(pipeline,f)

print("Pipeline trained and saved successfully.")