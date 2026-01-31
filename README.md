# Bank Consumer Churn Prediction

This project predicts bank consumer churn using a Random Forest Classifier. It demonstrates the use of MLflow for experiment tracking, model logging, and artifact management.

## Project Structure

The project is structured as follows:

- `dataset/`: Contains the dataset (`Churn_Modelling.csv`).
- `src/`: Contains the source code.
    - `preprocessing.py`: Functions for data loading, balancing, and preprocessing.
    - `model.py`: Functions for model training and evaluation.
    - `train.py`: Main script to run the training pipeline.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.8+
- MLflow
- Scikit-learn
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the MLflow server (optional if you just want to run the script, but `train.py` expects it at `http://localhost:5000`):

```bash
mlflow ui
```

2. Run the training script:

```bash
python src/train.py
```

The script will:
- Load and preprocess the data.
- Train a Random Forest model.
- Log metrics (accuracy, precision, recall, f1-score) and params to MLflow.
- Log artifacts (model, scaler, confusion matrix) to MLflow.

## Model Details

- **Model**: Random Forest Classifier
- **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target**: Exited (0 or 1)