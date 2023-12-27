#P2: Perform Data Loading, Feature selection (Principal Component analysis) and Feature Scoring and Ranking.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Saail Chavan 016")
#names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "species"]
dataset = pd.read_csv('/content/sample_data/iris.csv')
print(dataset.head(10))

X = dataset.drop(["species"], axis="columns")
Y = dataset["species"]

print(X)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print("Display Data  Including All Features")
model = PCA()
x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)
print(f"\nDataSet before PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")

print("Display Data Including Principal Features")
model = PCA(n_components=2)
x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)
print(f"\nDataSet After PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")
print("Saail Chavan 016")
