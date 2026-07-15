import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# load data
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# dependent variable
y = df ['species']

# independent variable
x = df.drop('species', axis=1)

print(df.head())

# split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# modesl
dt= DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SVC()

# train models
dt.fit(x_train, y_train)
rf.fit(x_train, y_train)
svm.fit(x_train, y_train)

# prediction
dt_pred = dt.predict(x_test)
rf_pred = dt.predict(x_test)
svm_pred = dt.predict(x_test)

# evaluation
print("Decision tree acccuracy", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("SVM accuracy: ", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


from sklearn.metrics import confusion_matrix
print("Decision Treee Confusion matrix is: ")
print(confusion_matrix(y_test, dt_pred))

print("Random Forest Accuracy: ")
print(confusion_matrix(y_test, rf_pred))

print("SVM Accuracy: ")
print(confusion_matrix(y_test, svm_pred))
