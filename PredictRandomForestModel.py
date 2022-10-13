#%%
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import pickle

from sklearn.ensemble import RandomForestClassifier

#read file
FILEPATH = 'source/loan_dataset.csv'
df = pd.read_csv(FILEPATH)


#Handle Categorical variables
df = df.drop(['Loan_ID'], axis = 1)
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

#Numerical Variables
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

#Transform variable that model can handle
df = pd.get_dummies(df)
# Drop columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)

#Loại bỏ các giá trị cực đoan (giá trị nằm rìa)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

#Chuyển về căn bậc 2 để tiện xử lí dữ liệu, nói chung học mới biết :DD
# Square Root Transformation
df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)

#print(df.head())
X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]

#Kĩ thuật smote, do có sự mất cân bằng giữa người được chấp nhận vay và người không được
X, y = SMOTE().fit_resample(X, y)

#Data Normalization
X = MinMaxScaler().fit_transform(X)

#Tách data để train thôi mấy mắm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Giờ thì vào model, t cũng đell hiểu đâu :D thấy ngta bảo model này ngon thì xài
#Random Forest
# scoreListRF = []
# for i in range(2,25):
#     RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
#     RFclassifier.fit(X_train, y_train)
#     scoreListRF.append(RFclassifier.score(X_test, y_test))
    
# RFAcc = max(scoreListRF)
# print("Random Forest Accuracy:  {:.2f}%".format(RFAcc*100))

#predict_model = RandomForestClassifier()
#predict_model.fit(X_train, y_train)

#load model
predict_model = joblib.load("source/Random_Forest.pkl")
predict_model_pickle = pickle.load(open("source/Random_Forest_p.sav", "rb"))

print(predict_model_pickle.predict(X_train[0].reshape(1, -1))[0])
#save model
#joblib.dump(predict_model, "source/Random_Forest.pkl")
#pickle.dump(predict_model, open("source/Random_Forest_p.sav", "wb"))
# %%
