def acc(thres):
    """testing the accuracy of a model using different thresholds"""
    temp = pd.DataFrame(lr_clf.predict_proba(X_test[feature_names])).max(axis=1) > thres
    accu = np.mean((predicted_lr == y_test).reset_index()['category'][temp[temp].index.tolist()])
    l = len(temp[temp].index.tolist())
    return  thres, accu, l


def acc2(thres, thres2):
    """testing the accuracy of a model using different confidence bands"""
    temp = (pd.DataFrame(lr_clf.predict_proba(X_test[feature_names])).max(axis=1) >= thres) & \
    (pd.DataFrame(lr_clf.predict_proba(X_test[feature_names])).max(axis=1) < thres2)
    accu = np.mean((predicted_lr == y_test).reset_index()['category'][temp[temp].index.tolist()])
    l = len(temp[temp].index.tolist())
    return  thres, accu, l

