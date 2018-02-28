import numpy as np
import cPickle
from sklearn.model_selection import StratifiedShuffleSplit


metas = cPickle.load(open('../metas.cPickle'))
X = metas.keys()
y = metas.values()

tr = []
val = []
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_idx, test_idx in sss.split(X, y):
    for item in train_idx:
        tr.append(X[item])

    for item in test_idx:
        val.append(X[item])

np.save(open('../tr', 'w'), tr)
np.save(open('../val','w'), val)

