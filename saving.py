import fire
import pickle
from preprocess import X_sc, y
from sklearn.ensemble import RandomForestClassifier

def save():
    RF = RandomForestClassifier(n_estimators=175, max_features=25, max_depth=29, criterion='entropy', n_jobs=-1)
    model = RF.fit(X_sc, y)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
fire.Fire()