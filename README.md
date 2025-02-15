# Cardiovascular Disease Prediction

This project aims to predict the presence of cardiovascular disease using machine learning techniques. The dataset used contains patient information such as age, gender, height, weight, blood pressure, cholesterol levels, and more. The goal is to build and evaluate machine learning models to accurately predict whether a patient has cardiovascular disease.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)

---

## Project Overview
Cardiovascular disease is one of the leading causes of death globally. Early detection and prediction of cardiovascular disease can help in timely intervention and treatment. This project uses a dataset containing patient health metrics to build machine learning models for predicting cardiovascular disease.

The project involves:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Training machine learning models (Logistic Regression, KNN, Random Forest)
- Evaluating model performance using accuracy, classification reports, and confusion matrices

---

## Dataset
The dataset used in this project is `cardio_train.csv`, which contains 70,000 rows and 13 columns. The features include:
- `age`: Patient age (in days)
- `gender`: Patient gender (1: female, 2: male)
- `height`: Patient height (in cm)
- `weight`: Patient weight (in kg)
- `ap_hi`: Systolic blood pressure
- `ap_lo`: Diastolic blood pressure
- `cholesterol`: Cholesterol level (1: normal, 2: above normal, 3: well above normal)
- `gluc`: Glucose level (1: normal, 2: above normal, 3: well above normal)
- `smoke`: Smoking status (0: non-smoker, 1: smoker)
- `alco`: Alcohol consumption (0: non-drinker, 1: drinker)
- `active`: Physical activity (0: inactive, 1: active)
- `cardio`: Presence of cardiovascular disease (0: no, 1: yes)

The dataset is available in the repository as `cardio_train.csv`.

---

## Requirements
To run this project, you need the following Python libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required libraries using the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iyedzarrougui14/Cardiovascular-Disease-Prediction
   ```
2. Navigate to the project directory:
   ```bash
   cd Cardiovascular-Disease-Prediction
   ```
3. Install the required libraries (see [Requirements](#requirements)).

---

## Usage
1. Open the Jupyter Notebook `Cardiovascular_Disease_Prediction.ipynb`.
2. Run the cells in the notebook to:
   - Load and preprocess the dataset
   - Perform exploratory data analysis (EDA)
   - Train machine learning models
   - Evaluate model performance
3. Modify the code or experiment with different models as needed.

---

## Results
The machine learning models were evaluated using accuracy, precision, recall, and F1-score. The results are as follows:
- **Logistic Regression**: Accuracy = X%
- **K-Nearest Neighbors (KNN)**: Accuracy = Y%
- **Random Forest**: Accuracy = Z%

For detailed results, refer to the Jupyter Notebook.

---

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## Acknowledgments
- Dataset: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

Feel free to reach out if you have any questions or suggestions!
```

### Key Sections:
1. **Project Overview**: A brief description of the project and its goals.
2. **Dataset**: Information about the dataset and its features.
3. **Requirements**: Lists the Python libraries needed to run the project.
4. **Installation**: Steps to clone the repository and install dependencies.
5. **Usage**: Instructions on how to run the Jupyter Notebook.
6. **Results**: Summary of model performance.
7. **Contributing**: Guidelines for contributing to the project.
