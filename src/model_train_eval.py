import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    # Define models
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    result_selection = {
        'Model': [],
        'Score': [],
        'RMSE': []
    }

    # Multiple Linear Regression
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test) 
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Multiple Linear Regression')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    #Ridge
    regressor = Ridge(alpha=1)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Ridge')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    #Polynomial Degree = 2
    from math import degrees
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly,y_train)
    y_pred = regressor.predict(poly_reg.transform(x_test))
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Polynomial(degree = 2)')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    #Polynomial Degree = 3
    poly_reg = PolynomialFeatures(degree=3)
    x_poly = poly_reg.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly,y_train)
    y_pred = regressor.predict(poly_reg.transform(x_test))
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Polynomial(degree = 3)')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    # SVR
    regressor = SVR()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Support Vector Regression')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    # Decision Tree
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Support Vector Regression')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    # Random Forest
    regressor = RandomForestRegressor(n_estimators=15, random_state=42)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('Random Forest (n = 15)')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    # ANN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=12,activation='relu'))
    model.add(tf.keras.layers.Dense(units=12,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1,activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)
    y_pred = model.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('ANN')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)


    # XGBoost
    regressor = model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=5, n_estimators=15, random_state=42)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    result_selection['Model'].append('XGBoost')
    result_selection['Score'].append(score)
    result_selection['RMSE'].append(rmse)
    
    #Analysing Result
    analysis = pd.DataFrame(result_selection)
    img = analysis.to_csv('analysis.csv')
    best_model = analysis.loc[analysis['Score'].idxmax(), 'Model']
    
    #grid search for best model
    regressor = RandomForestRegressor()
    n = [i for i in range(5,51)]
    params = {
        'n_estimators': n,
        'random_state' : [42]
    }
    grid = GridSearchCV(estimator=regressor, param_grid=params, cv=10)
    grid.fit(x_train,y_train)
    y_pred = grid.predict(x_test)
    score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    para = grid.best_params_
    
    # Feature importance of best model
    regressor = RandomForestRegressor(n_estimators=26, random_state=42)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    importances = regressor.feature_importances_
    
    #Feature Importance Plot
    df = pd.read_excel('data/Data.xlsx')
    df = df.drop('hardness',axis=1)
    feature_names = np.array(df.columns)  # Replace with your actual feature names
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(x_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(x_train.shape[1]), feature_names[indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()