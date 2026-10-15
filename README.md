# Heart Disease Prediction

A production-style machine learning project designed to predict the likelihood of heart disease using patient health indicators. This repository demonstrates a complete data science workflow, including data exploration, feature analysis, model training, and evaluation in a Jupyter Notebook environment.

## Overview

Heart disease remains one of the leading causes of mortality worldwide. Early detection and risk assessment can significantly improve patient outcomes. This project uses commonly available clinical attributes such as age, sex, blood pressure, cholesterol, heart rate, and other health indicators to train a predictive model.

The goal is to provide a clear, reproducible workflow for:

- loading and inspecting medical data
- exploring feature relationships
- understanding class distribution
- training a machine learning classifier
- evaluating model performance

## Project Structure

```text
Heart_Disease_Prediction/
├── Heart_disease_prediction (1).ipynb   # Main notebook with data analysis and model development
├── heart.csv                             # Dataset used for training and evaluation
├── README.md                            # Project documentation
└── .ipynb_checkpoints/                  # Jupyter checkpoint files (auto-generated)
```

## Features

- Data preprocessing and exploratory data analysis (EDA)
- Health feature interpretation and correlation analysis
- Classification model training using Python and scikit-learn
- Visual insights with Matplotlib and Seaborn
- Clear notebook-based workflow for learning and experimentation

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses a heart disease dataset containing clinical features such as:

- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- fasting blood sugar
- resting ECG results
- maximum heart rate achieved
- exercise-induced angina
- oldpeak
- slope
- number of major vessels
- thalassemia status
- target variable (heart disease presence/absence)

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### Clone the Repository

```bash
git clone https://github.com/Sai-Niresh/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook
```

Open the notebook named:

```text
Heart_disease_prediction (1).ipynb
```

## Workflow

1. Import required libraries
2. Load the dataset
3. Inspect the data structure and quality
4. Perform exploratory data analysis
5. Study relationships between features and the target variable
6. Train a classification model
7. Evaluate performance using metrics and confusion matrix
8. Interpret results and draw conclusions

## Model Evaluation

The notebook focuses on evaluating a classification model for heart disease prediction using standard metrics such as:

- accuracy
- confusion matrix
- feature importance or correlation insights

This provides a practical understanding of how the model performs and which patient characteristics contribute most to prediction.

## Example Requirements

If you want to make this project more production-ready, a `requirements.txt` file can be added with dependencies such as:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Production-Ready Recommendations

To make this project more suitable for deployment or team use, consider the following improvements:

- add a `requirements.txt` file
- create a reusable Python script for training and prediction
- separate data preprocessing and model logic into modules
- store the trained model artifact using joblib or pickle
- add unit tests for data validation and prediction pipeline
- build a simple API or dashboard for real-time inference

## Contribution

Contributions are welcome. If you would like to improve the project, please fork the repository, create a feature branch, and submit a pull request.

## Author

Sai Niresh Narayan

## License

This project is available for educational and research use.

## Acknowledgements

This project is inspired by the broader goal of using machine learning for early disease detection and healthcare decision support.
