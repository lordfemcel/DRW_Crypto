from sklearn.linear_model import SGDRegressor

def augment_with_xgb_preds(X, preds):
    X_aug = X.copy()
    X_aug["xg_pred"] = preds
    return X_aug

def train_sgd(X_train, y_train):
    sgd = SGDRegressor(max_iter=500, tol=1e-3, random_state=42)
    sgd.fit(X_train, y_train)
    return sgd
