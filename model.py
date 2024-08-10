import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_train_eval import train_and_evaluate_models

class My_Model:

    def __init__(self):
        self.model = self.build_model()
        
    def build_model(self):
        # Define file path for data
        data_file_path = 'data/Data.xlsx'
    
        # Load and preprocess data
        x_train, x_test, y_train, y_test = preprocess_data(data_file_path)
    
        # Train and evaluate models
        train_and_evaluate_models(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    my_model = My_Model()