import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
import sklearn.tree as tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

"""

Name:(Kyle Webb)
Date:(12/4/24)
Assignment:(Project 4)
Due Date:(12/1/24)
About this project:(Regression Models to predict housing prices)
Assumptions:(None)
All work below was performed by (Kyle Webb)

"""


import matplotlib
matplotlib.use('TkAgg')  # trouble with pycharm to load graph...

def ShowKNNRegression(prices,X,y):
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = neighbors.KNeighborsRegressor(n_neighbors=4)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("KNN Regression")

    r2 = r2_score(y_test, predictions)
    print(f'R^2: {r2:.2f}')

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R²: {scores.mean():.2f}")

    print('Index\tPredicted\tActual')
    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            print(i, '\t', predictions[i], '\t', y_test[i])

def ShowLinearRegression(prices,X,y):
    model = LinearRegression(fit_intercept=False)
    clf = model.fit(X, y)

    predictions = model.predict(X)

    print("Linear Regression")

    r2 = r2_score(y, predictions)
    print(f'R^2: {r2:.2f}')

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R²: {scores.mean():.2f}")

    for index in range(len(predictions)):
        if float(y[index]) != float(predictions[index]):
            print('Actual Price:', y[index], 'Predicted Price:', predictions[index])

def ShowDecisionTreeRegression(prices,X,y):
    X = np.array(X)
    y = np.array(y)

    # split the data into the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Decision Tree Regression")

    r2 = r2_score(y_test, predictions)
    print(f'R^2: {r2:.2f}')

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R²: {scores.mean():.2f}")

    print('Index\tPredicted\tActual')
    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            print(i, '\t', predictions[i], '\t', y_test[i])


def ShowDecisionTreeForestRegression(prices,X,y):
    X = np.array(X)
    y = np.array(y)

    # split the data into the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(model.feature_importances_)
    print("Decision Tree Forest Regression")

    r2 = r2_score(y_test, predictions)
    print(f'R^2: {r2:.2f}')

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R²: {scores.mean():.2f}")

    print('Index\tPredicted\tActual')
    for i in range(len(predictions)):
        if predictions[i] != y_test[i]:
            print(i, '\t', predictions[i], '\t', y_test[i])
    # Extract single tree
    estimator = model.estimators_[0]

def main():
    '''
    Prediction (40 points) -- Housing Prices
    Source: https://www.kaggle.com/datasets/fratzcan/usa-house-prices
    '''

    # Question: What is the price of the house given features of the home.
    prices = pd.read_csv('USA Housing Dataset.csv')
    print(prices.columns)

    # Compute correlation matrix
    corr_matrix = prices[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built']].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    #plt.show()

    X = prices[['bathrooms', 'bedrooms', 'waterfront', 'view', 'sqft_living', 'sqft_above','yr_built']].values
    y = prices['price']

    ShowLinearRegression(prices,X,y)
    ShowDecisionTreeRegression(prices,X,y)
    ShowDecisionTreeForestRegression(prices,X,y)
    ShowKNNRegression(prices,X,y)




if __name__ == "__main__":
    main()