#P4: For a given set of training data examples stored in a .CSV file implement Least Square Regression algorithm. (Use Univariate dataset)
#Dataset: Diabetes.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

print("Saail Chavan 016")
dataset = pd.read_csv("/content/sample_data/diabetes.csv")
print(dataset.head())

x = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, [-1]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n ", cm)

print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
print("Saail Chavan 016")
