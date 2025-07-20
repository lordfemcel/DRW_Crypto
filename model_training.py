import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def optuna_objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "tree_method": "gpu_hist",
        "device": "cuda",
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_leaves": trial.suggest_int("max_leaves", 8, 64),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 50.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 80.0),
        "n_estimators": 1667
    }

    model = xgb.XGBRegressor(early_stopping_rounds=50, **params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    preds = model.predict(X_valid)
    r, _ = pearsonr(y_valid, preds)
    return r

def run_optuna(X_train, y_train, X_valid, y_valid, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_valid, y_valid), n_trials=n_trials)
    return study.best_params, study.best_value

def train_xgb(X_train, y_train, X_valid, y_valid, best_params):
    best_params.update({
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse"
    })
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    return model
