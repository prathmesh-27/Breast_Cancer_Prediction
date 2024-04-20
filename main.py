import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score,accuracy_score
import pandas as pd
import joblib

# loading dataset
data=load_breast_cancer()
cancer_features=pd.DataFrame(data.data,columns=data.feature_names)


#removing not relevant features
imp_cancer_features = cancer_features.drop(['mean perimeter','mean area','mean radius','mean compactness'],axis=1)

## Label encoding using the LabelEncoder of Scikit-learn

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

## Converting the data type of the variables to pandas series
target_series = pd.Series(data.target)
target = le.fit_transform(target_series)

#Standard Scaler
STD = StandardScaler()
imp_cancer_features = STD.fit_transform(imp_cancer_features)


#Data Reduction using  Principal Component Analysis (PCA)
pca = PCA(n_components=10)
fit = pca.fit(imp_cancer_features)
pca_cancer_features = pca.transform(imp_cancer_features)

# split data
x_train,x_test,y_train,y_test=train_test_split(pca_cancer_features,target,test_size=0.3,random_state=0)

print("Train Data Shape: ", x_train.shape)
print("Train Target Shape: ", y_train.shape)
print("Test Data Shape: ", x_test.shape)
print("Test Target Shape: ", y_test.shape)

#model selection
model=SVC(C=1.2,kernel='rbf')

model.fit(x_train,y_train)

y_pred = model.predict(x_test)



print("accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))
print("area under curve (auc): ", roc_auc_score(y_test, y_pred))


# Save both the vectorizer and the model in a single file
model_path = os.getcwd() + r'\models\model\cancer_predictor.pkl'
joblib.dump((STD,pca,model), model_path, compress=True)

print('Model successfully extracted and pickled.')
