import joblib
import numpy as np
import pandas as pd

# Load your model, feature names, and scaler
model = joblib.load('random_forest_model.pkl')
feature_names = joblib.load('features.pkl')
sc = joblib.load('scaler.pkl')

# Function to load CSV and Excel files
def load_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(data.head())  # Display the first few rows
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        return None

def load_excel_file(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Data loaded successfully!")
        print(data.head())  # Display the first few rows
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        return None

def main():
    while True:
        print("Choose one of the options:")
        print("1. Dataset")
        print("2. By giving input for each feature")
        choice = input("Enter 1 or 2: ")

        if choice == '2':
            feature_values = []
            for feature in feature_names:
                value = float(input(f"Enter the value for {feature}: "))
                feature_values.append(value)

            input_data = np.array(feature_values).reshape(1, -1)
            input_data = sc.transform(input_data)
            prediction = model.predict(input_data)
            print(f"The predicted class is: {prediction[0]}")
        
        elif choice == '1':
            print("NOTE: Features in dataset should be in the given format.")
            print(feature_names)
            print("Choose type of file:")
            print("1. CSV")
            print("2. Excel")
            file_type = input("Enter 1 or 2: ")

            file_path = input("Enter the path to the file: ")

            if file_type == '1':
                df = load_csv_file(file_path)
            elif file_type == '2':
                df = load_excel_file(file_path)
            else:
                print("Invalid file type chosen.")
                continue

            if df is not None:
                x_test = df.iloc[:,:-1].values
                x_test = sc.transform(x_test)
                y_pred = model.predict(x_test)
                y_pred = y_pred.reshape(len(y_pred), 1)
                print(f"Predicted Hardness are: {y_pred.flatten()}")
                # Save predictions to a CSV file
                pd.DataFrame(y_pred, columns=['Predicted Hardness']).to_csv("Predicted_Hardness.csv", index=False)
                print("Predictions saved to Predicted_Hardness.csv")
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

        repeat = input("Do you want to try again? (yes/no): ").strip().lower()
        if repeat != 'yes':
            break

if __name__ == "__main__":
    main()
