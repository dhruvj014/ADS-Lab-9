import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import LocalOutlierFactor
df=pd.read_csv(r"C:\Users\MEET\Desktop\Stellar project\star_classification.csv")
df.drop(columns=['alpha','delta',"obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(df) 
x_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score
threshold = np.quantile(x_score , .10)                                            
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()
df.drop(outlier_index, inplace=True)
x = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
X = scaler.transform(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
final_model = RandomForestClassifier()
final_model.fit(X_train, y_train)
y_pred1 = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred1)
print('Accuracy:', accuracy)
pickle.dump(final_model,open("model.pkl","wb"))
