import numpy as np
import pandas as pd
print("Data read")
train_data = pd.read_csv("/home/nikhil/Loan Delinquency Prediction/train_u5jK80M/train.csv")
test_data = pd.read_csv("/home/nikhil/Loan Delinquency Prediction/test_3BA6GZX/test.csv")
#print(train_data.head()
ss = train_data[train_data["m13"]==1]
dd = ss[train_data["loan_to_value"]<20]
aa = dd.index
#Detect Outlier
train_data = train_data.drop(train_data.index[[45,507,621,633,107296]])

target = train_data.m13
test_Id = test_data.loan_id

train_data = train_data.drop("m13",axis=1)
combined_data = pd.concat([train_data,test_data],axis=0)
combined_data = combined_data.drop("loan_id",axis=1)
from sklearn.preprocessing import LabelEncoder
leb = LabelEncoder()
combined_data["source"] = leb.fit_transform(combined_data["source"])
combined_data["financial_institution"] = leb.fit_transform(combined_data["financial_institution"])
combined_data["origination_date"] = leb.fit_transform(combined_data["origination_date"])
combined_data["first_payment_date"] = leb.fit_transform(combined_data["first_payment_date"])
combined_data["loan_purpose"] = leb.fit_transform(combined_data["loan_purpose"])

#==============Feature Engineering==============#
#EMI = [P x R x (1+R)^N]/[(1+R)^N-1] 
combined_data["EMI"] =  (combined_data["unpaid_principal_bal"]*(combined_data["interest_rate"]/100.0)*(1+combined_data["interest_rate"]/100.0)**combined_data["loan_term"])/((1+combined_data["interest_rate"]/100.0)**(combined_data["loan_term"]-1))
combined_data["New_feature1"] = (combined_data["loan_to_value"]+combined_data["debt_to_income_ratio"])/combined_data["loan_term"]
combined_data["New_feature2"] = 0.8*combined_data["borrower_credit_score"]+0.2*combined_data["co-borrower_credit_score"]

#==============Drop Unimportant Feature on the basis of feature importance of Random Forest==========#
combined_data.drop(["origination_date","first_payment_date","source","financial_institution"],axis=1,inplace=True)
print("preprocessing done")
ttrain_data = combined_data.iloc[0:116053,:]
ttest_data = combined_data.iloc[116053::,:]

#=============Oversampling technique==============#
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X,Y = sm.fit_sample(ttrain_data.values,target.values)
from sklearn.utils import shuffle
for i in range(200):
    feature,label = shuffle(X,Y,random_state=0)

print("oversampling done")

#============Import Model==========================#
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,make_scorer
score = make_scorer(f1_score)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
clf3 = GaussianNB()
rnd_clf = RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)
extree_tree_clf = ExtraTreesClassifier(n_estimators=300,random_state=42,n_jobs=-1)
ada_clf = AdaBoostClassifier(extree_tree_clf,n_estimators= 300,algorithm="SAMME.R",learning_rate=0.01)
mlp_clf = MLPClassifier(random_state=42)
model_LGBM =lgb.LGBMClassifier(learning_rate=0.01,objective='binary')

#===============Voting Classifier==================#
from sklearn.ensemble import VotingClassifier
#named_estimators = [("ada_clf",ada_clf),("rnd_clf",rnd_clf),("extree_tree_clf",extree_tree_clf)]
#voting_clf = VotingClassifier(named_estimators,voting="soft",flatten_transform=True)
#voting_clf.fit(feature,label)
#test_result = pd.concat([test_Id,pd.DataFrame(result)],axis=1)
#test_result.columns = ["loan_id","m13"]
#test_result.to_csv("loan_delinqncy20.csv", encoding='utf-8', index=False)

#===============Random Forest Classifier============#
rnd_clf = RandomForestClassifier(max_depth = 10,n_estimators=500,random_state=42,n_jobs=-1)
rnd_clf.fit(feature,label)
result = rnd_clf.predict(ttest_data)
test_result = pd.concat([test_Id,pd.DataFrame(result)],axis=1)
test_result.columns = ["loan_id","m13"]
test_result.to_csv("loan_delinqncy500.csv", encoding='utf-8', index=False)

