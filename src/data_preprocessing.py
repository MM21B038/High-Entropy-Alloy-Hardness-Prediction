import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Separate features and target variable
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    return x_train, x_test, y_train, y_test