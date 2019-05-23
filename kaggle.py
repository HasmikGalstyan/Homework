import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train-3.csv')
test = pd.read_csv('test-3.csv')

oof = np.zeros(len(train))
preds = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

for i in range(512):

    train2 = train[train['wheezy-copper-turtle-magic'] == i]
    test2 = test[test['wheezy-copper-turtle-magic'] == i]
    idx1 = train2.index;
    idx2 = test2.index
    train2.reset_index(drop=True, inplace=True)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])

    skf = StratifiedKFold(n_splits=13, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        clf = KNeighborsClassifier(weights='distance')
        clf.fit(train3[train_index, :], train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index, :])[:, 1]
        preds[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print('CV score =', round(auc, 5))

sub = pd.read_csv('sample_submission.csv')
sub['target'] = preds
sub.to_csv('sample_submission.csv',index=False)
