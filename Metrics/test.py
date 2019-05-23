from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from metrics import Metrics


if __name__ == '__main__':
    data = datasets.load_breast_cancer().data
    target = datasets.load_breast_cancer().target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

    rf = RandomForestClassifier(n_estimators=10, max_depth=2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)[:, 1]

    print('ROC_curve')
    print(Metrics().roc(y_test, y_pred))
    print('ROC_AUC_score')
    ras = Metrics().roc_auc(y_test, y_pred)
    print(ras)
    Metrics().plot(y_test, y_pred, ras)
