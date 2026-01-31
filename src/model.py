"""
This module contains functions to train and evaluate the model.
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow

def train(X_train, y_train):
    """
    Train a Random Forest model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestClassifier: trained random forest model
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    input_schema = mlflow.models.infer_signature(X_train, clf.predict(X_train))
    # output_schema = mlflow.models.infer_signature(y_train, clf.predict(X_train)) # Not strictly needed if signature covers input/output


    # Log model
    mlflow.sklearn.log_model(clf, "random_forest_model", signature=input_schema)

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")

    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and log metrics to MLflow.

    Args:
        model (LogisticRegression): trained model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test target
    """
    y_pred = model.predict(X_test)

    ### Log metrics after calculating them
    accuracy = accuracy_score(y_test, y_pred)

    ### Log tag
    mlflow.set_tag("model_type", "random_forest")

    ### Log metrics
    mlflow.log_metric("accuracy", accuracy)
    precision = precision_score(y_test, y_pred)
    mlflow.log_metric("precision", precision)
    recall = recall_score(y_test, y_pred)
    mlflow.log_metric("recall", recall)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)

    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    # Log the image as an artifact in MLflow
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # plt.show()