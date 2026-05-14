"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import joblib
import pandas as pd
import mlflow
import preprocessing
import model


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    mlflow.set_experiment("Bank Consumer Churn Prediction")


    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="random_forest_run"):

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocessing.preprocess(df)


        trained_model = model.train(X_train, y_train)

        model.evaluate_model(trained_model, X_test, y_test)

        ### Persist the best model locally so the API can load it without MLflow
        joblib.dump(trained_model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        print("Saved model.pkl to repo root.")


if __name__ == "__main__":
    main()
