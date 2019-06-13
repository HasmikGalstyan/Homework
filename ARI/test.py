import pandas as pd
import numpy as np
from ari import ARI

train = pd.read_csv('ts_train.csv', sep=",")
test = pd.read_csv('ts_test.csv')

predicted = np.array([])
for i in range(1,23):
    ari = ARI(s=[7, 365], d=2, p=1)
    ari.train(np.array(train[train.tsID == i].ACTUAL))
    predicted = np.append(predicted,ari.predict(num=300))

ss = pd.concat([pd.Series(test['ID'], name='id'),pd.Series(predicted,name='value')], axis=1)
ss.to_csv('sample_submission.csv',index=False,header=True)