import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm
import seaborn as sns
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
    models_analysis = analysis.to_csv('analysis.csv')
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
    
    df = pd.read_excel('data/Data.xlsx')
    #Feature Importance Plot
    feature_names = df.columns
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = feature_names[indices]

   # Normalize the importances for color mapping
    norm = Normalize(vmin=sorted_importances.min(), vmax=sorted_importances.max())
    colors = LinearSegmentedColormap.from_list("blue_darkblue", ["#87CEFA", "#00008B"])
    mapped_colors = colors(norm(sorted_importances))
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(sorted_importances)), sorted_importances, color=mapped_colors, align='center')
    plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.savefig('feature_importance.png')
    plt.show()

    # 1. Reduced Modulus vs Hardness
    plt.figure(figsize=(10, 6))
    plt.scatter(sc.inverse_transform(x_train)[:, 7], y_train, label='Train Set', alpha=0.7) # Inverse transform the entire training set and extract the first column
    plt.scatter(sc.inverse_transform(x_test)[:, 7], y_test, label='Test Set', alpha=0.7) # Inverse transform the entire test set and extract the first column
    plt.scatter(sc.inverse_transform(x_test)[:, 7], y_pred, label='Predictions', marker='x', color='red') # Inverse transform the entire test set and extract the first column
    plt.xlabel('Reduced Modulus')
    plt.ylabel('Hardness')
    plt.title('Reduced Modulus vs Hardness')
    plt.legend()
    plt.grid(True)
    plt.savefig('Reduced_Modulus_vs_Hardness.png')
    plt.show()

    # 2. Y vs Hardness
    plt.figure(figsize=(10, 6))
    plt.scatter(sc.inverse_transform(x_train)[:, 6], y_train, label='Train Set', alpha=0.7) # Inverse transform the entire training set and extract the second column
    plt.scatter(sc.inverse_transform(x_test)[:, 6], y_test, label='Test Set', alpha=0.7) # Inverse transform the entire test set and extract the second column
    plt.scatter(sc.inverse_transform(x_test)[:, 6], y_pred, label='Predictions', marker='x', color='red') # Inverse transform the entire test set and extract the second column
    plt.xlabel('Y')
    plt.ylabel('Hardness')
    plt.title('Y vs Hardness')
    plt.legend()
    plt.grid(True)
    plt.savefig('Y_vs_Hardness.png')
    plt.show()

    # 3. 3D graph with x-axis = reduced modulus, y-axis = Y, z-axis = hardness
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sc.inverse_transform(x_train)[:,7], sc.inverse_transform(x_train)[:,6], y_train, label='Train Set')
    ax.scatter(sc.inverse_transform(x_test)[:,7], sc.inverse_transform(x_test)[:,6], y_test, label='Test Set')
    ax.scatter(sc.inverse_transform(x_test)[:,7], sc.inverse_transform(x_test)[:,6], y_pred, c='red', marker='x', label='Predictions')
    ax.set_xlabel('Reduced Modulus')
    ax.set_ylabel('Y')
    ax.set_zlabel('Hardness')
    plt.title('3D Scatter Plot: Reduced Modulus, Y, Hardness')
    plt.savefig('3d_scatter_plot_Reduced_Modulus_Y_Hardness.png')
    plt.show()

   # 4. 3D graph with x-axis = W, y-axis = Ta, z-axis = hardness
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sc.inverse_transform(x_train)[:,3], sc.inverse_transform(x_train)[:,2], y_train, label='Train Set')
    ax.scatter(sc.inverse_transform(x_test)[:,3], sc.inverse_transform(x_test)[:,2], y_test, label='Test Set')
    ax.scatter(sc.inverse_transform(x_test)[:,3], sc.inverse_transform(x_test)[:,2], y_pred, c='red', marker='x', label='Predictions')
    ax.set_xlabel('W')
    ax.set_ylabel('Ta')
    ax.set_zlabel('Hardness')
    plt.title('3D Scatter Plot: W, Ta, Hardness')
    plt.savefig('3d_scatter_plot_W_Ta_Hardness.png')
    plt.show()

    # sigma_T vs Hardness
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x="sigma_T", y="hardness", data=df, palette="PuBu")
    plt.title("sigma_T vs Hardness")
    plt.ylim(4,11)
    plt.savefig('sigma_T_vs_Hardness.png')
    plt.show()

    # Predicted vs. Actual Values
    residuals = y_pred - y_test
    abs_residuals = np.abs(residuals)
    norm_residuals = abs_residuals / np.max(abs_residuals)
    cmap = plt.cm.get_cmap("PuBu")  # You can choose any colormap you prefer
    colors = cmap(norm_residuals)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, hue=norm_residuals, palette="PuBu", legend=False, edgecolor=None)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.max(abs_residuals)))
    sm.set_array([])
    plt.colorbar(sm, label='Distance from Diagonal Line')
    plt.savefig('Predicted_vs_Actual.png')
    plt.show()

    # Residuals Distribution
    residuals = y_test - y_pred
    counts, bins = np.histogram(residuals, bins=30)
    norm = Normalize(vmin=counts.min(), vmax=counts.max())
    cmap = plt.cm.PuBu
    plt.figure(figsize=(8, 5))
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
    	plt.fill_between([left, right], 0, count, color=cmap(norm(count)))
    sns.kdeplot(residuals, color='black')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('residuals_distribution.png')
    plt.show()
