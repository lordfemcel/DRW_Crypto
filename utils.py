import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def kfold_oof_xgb(X, y, params, n_splits=5):
    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
                  verbose=False)
        oof_preds[valid_idx] = model.predict(X.iloc[valid_idx])
    return oof_preds

def scale_features(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler
