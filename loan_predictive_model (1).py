# import libraries
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

# handling null values
data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Loan_Amount_Term']=data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean())
data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mean())
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# Data Visulization

fig, ax =plt.subplots(3,2)
fig.tight_layout()
sns.countplot(data['Gender'], ax=ax[0,0])
sns.countplot(data['Married'], ax=ax[0,1])
sns.countplot(data['Self_Employed'], ax=ax[1,0])
sns.countplot(data['Credit_History'], ax=ax[1,1])
sns.countplot(data['Education'], ax=ax[2,0])
sns.countplot(data['Property_Area'], ax=ax[2,1])

plt.show()



# handling categorical values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
data['Married'] = encoder.fit_transform(data['Married'])
data['Dependents'] = encoder.fit_transform(data['Dependents'])
data['Education'] = encoder.fit_transform(data['Education'])
data['Self_Employed'] = encoder.fit_transform(data['Self_Employed'])
data['Property_Area'] = encoder.fit_transform(data['Property_Area'])
data['Loan_Status'] = encoder.fit_transform(data['Loan_Status'])






# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X, y = data.iloc[:, 1:12], data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy = {:.2f}%".format(classifier.score(X_test, y_test)*100))

y_pred1 = classifier.predict(X)
print(y_pred1)

# save model
filename = 'predictive_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


