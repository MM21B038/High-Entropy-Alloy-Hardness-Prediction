import joblib
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

# Load your model, feature names, and scaler
model = joblib.load('random_forest_model.pkl')
feature_names = joblib.load('features.pkl')
sc = joblib.load('scaler.pkl')

# Initialize a global variable for dataframe
df = None

# Define functions to load CSV and Excel files
def load_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        display(data)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        return None

def load_excel_file(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Data loaded successfully!")
        display(data)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        return None

# Function to handle file upload and process data
def on_load_button_clicked(change):
    global df
    if location.value:
        for name, file_info in location.value.items():
            with open(name, 'wb') as f:
                f.write(file_info['content'])
            if file_type.value == 'CSV':
                df = load_csv_file(name)
            elif file_type.value == 'Excel':
                df = load_excel_file(name)
                
        if df is not None:
            x_test = df.iloc[:, :].values
            x_test = sc.transform(x_test)
            y_pred = model.predict(x_test)
            y_pred = y_pred.reshape(len(y_pred), 1)
            print(f"Predicted Hardness are: {y_pred.flatten()}")
            # Save predictions to a CSV file
            pd.DataFrame(y_pred, columns=['Predicted Hardness']).to_csv("Predicted_Hardness.csv", index=False)
            print("Predictions saved to Predicted_Hardness.csv")

# Setup widgets for file type and file upload
file_type = widgets.Dropdown(
    options=['CSV', 'Excel'],
    value='CSV',
    description='Choose type of file:',
    disabled=False
)
location = widgets.FileUpload(accept='', multiple=False)
load_button = widgets.Button(description="Load File")

# Display widgets
display(file_type)
display(location)
display(load_button)

# Link the load button to the function
load_button.on_click(on_load_button_clicked)

# Dropdown for selecting the mode of input
choose = widgets.Dropdown(
    options=['Dataset', 'By giving input for each feature'],
    value='Dataset',
    description='Choose one of the option:',
    disabled=False
)

def handle_mode_selection(change):
    if choose.value == 'By giving input for each feature':
        feature_values = []
        for feature in feature_names:
            value = float(input(f"Enter the value for {feature}: "))
            feature_values.append(value)

        input_data = np.array(feature_values).reshape(1, -1)
        input_data = sc.transform(input_data)
        prediction = model.predict(input_data)
        print(f"The predicted class is: {prediction[0]}")
    else:
        print("NOTE: Features in dataset should be in the given format.")
        print(feature_names)
        display(file_type)
        display(location)
        display(load_button)

display(choose)
choose.observe(handle_mode_selection, names='value')

# Handle repetition
repeat = widgets.Dropdown(
    options=['Yes', 'No'],
    value='Yes',
    description='Do you want to try again:',
    disabled=False
)

def handle_repeat_selection(change):
    global re
    if repeat.value == 'Yes':
        re = True
    else:
        re = False

display(repeat)
repeat.observe(handle_repeat_selection, names='value')
