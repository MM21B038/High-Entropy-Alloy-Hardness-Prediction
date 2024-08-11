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
   cd High-Entropy-Alloy-Hardness-Prediction

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run the project:
   ```bash
   python model.py

## Usage

1. Data Preprocessing: Run src/data_preprocessing.py to preprocess the data.
2. Model Training and Evaluation: Run src/model_train_eval.py to train models.

## Dataset Description
The dataset consists of 311 entries with 20 features, primarily focusing on the composition and properties of high-entropy alloys (HEAs) and their hardness values. Each feature in the dataset is a critical factor that contributes to the understanding and prediction of alloy hardness. Below is a description of each feature:

1. `Mo`: Molybdenum composition in the alloy (percentage).
2. `Nb`: Niobium composition in the alloy (percentage).
3. `Ta`: Tantalum composition in the alloy (percentage).
4. `W`: Tungsten composition in the alloy (percentage).
5. `sigma_T`: Statistical parameter related to the electronic structure.
6. `x`: Feature representing some material characteristic (specific meaning not provided).
7. `y`: Feature representing some material characteristic (specific meaning not provided).
8. `Reduced modulus`: A measure of the material's stiffness.
9. `E_ROM`: Young's modulus is predicted from the rule of mixtures (ROM).
10. `C11_ROM`: Predicted C11 elastic constant from ROM.
11. `C12_ROM`: Predicted C12 elastic constant from ROM.
12. `C44_ROM`: Predicted C44 elastic constant from ROM.
13. `Shear modulus from ROM`: Predicted shear modulus from ROM.
14. `Bulk modulus from ROM`: Predicted bulk modulus from ROM.
15. `V_ROM`: Volume of the unit cell predicted from ROM.
16. `dV_ROM`: Volume change predicted from ROM.
17. `HV_ROM`: Hardness predicted from ROM.
18. `a_ROM`: Lattice parameter is predicted from ROM.
19. `b_ROM`: Lattice parameter b predicted from ROM.
20. `Hardness`: Measured hardness of the alloy (target variable).
    
This dataset is essential for developing models that can predict the hardness of high-energy alloys based on their composition and various physical properties. The target variable is the `"Hardness"` column, which we aim to predict using the other features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
