# High-Entropy-Alloy-Hardness

## Overview

This project focuses on predicting the hardness of high-entropy alloy Mo-Nb-Ta-W using various machine learning models. The models evaluated include Linear Regression, Ridge Regression, Polynomial Regression, Support Vector Regression, Decision Trees, Random Forests, Neural Networks, and XGBoost.

## Directory Structure

- `data/`: Contains the dataset.
- `src/`: Contains the source code for data preprocessing, model training and evaluation.
- `model.py`: Contains the final model definition and training script.
- `requirements.txt`: Lists the required Python packages.
- `setup.py`: For setting up the package.
- `Dockerfile` and `docker-compose.yml`: For containerizing the application.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MM21B038/High-Entropy-Alloy-Hardness-Prediction.git
   cd High-Entropy-Alloy-Hardness

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run the project:
   ```bash
   python model.py

## Usage

1. Data Preprocessing: Run src/data_preprocessing.py to preprocess the data.
2. Model Training and Evaluation: Run src/model_train_eval.py to train models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.