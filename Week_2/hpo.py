import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
#@click.option(
#    "--num_trials",
#    default=15,
#    help="The number of parameter evaluations for the optimizer to explore"
#)
def run_optimization(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
    
        with mlflow.start_run():
            mlflow.set_tag("developer", "Krishna")
            mlflow.set_tag("model", "rf-regressor")
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_param("n_estimators",params['n_estimators'])
            mlflow.log_param("max_depth",params['max_depth'])
            mlflow.log_param("min_samples_split",params['min_samples_split'])
            mlflow.log_param("min_samples_leaf",params['min_samples_leaf'])
            mlflow.log_artifacts("./artifacts")

        return {"loss": rmse, "status": STATUS_OK}
    
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()
