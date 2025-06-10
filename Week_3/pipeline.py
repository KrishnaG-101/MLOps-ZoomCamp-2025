from datetime import timedelta
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy import sparse
import mlflow
from prefect import flow, task
from prefect.tasks import task_input_hash
from pathlib import Path

DATA_PATH = './data/yellow_tripdata_2023-03.parquet'

@task
def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

@task
def transform_data(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    print(f"The transformed dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

@task
def prepare_data(df):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    y = df["duration"].to_numpy()
    return X, y

@task
def train_model(X_train, y_train):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc_taxi_experiment")
    
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.log_param("intercept_", lr.intercept_)
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="artifacts_local",
            registered_model_name="MyLinearRegressor"
        )

@flow(name="Taxi ML Pipeline", retries=1, retry_delay_seconds=300)
def taxi_pipeline():
    df = load_data()
    df = transform_data(df)
    X, y = prepare_data(df)
    train_model(X_train=X, y_train=y)

if __name__ == "__main__":
    taxi_pipeline()