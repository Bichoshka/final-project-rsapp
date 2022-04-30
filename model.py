from comet_ml import Experiment

experiment = Experiment(
    api_key="lpFixqrYj9JT21PTd3XaoHgHm",
    project_name="general",
    workspace="bichoshka",
)

from preprocess import *
#import Forest as f
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=451)
final = []
def low_fit(model):
    model.fit(x_train_sc, y_train) #фит

    y_pred=model.predict(x_test_sc)#предсказание

    score=accuracy_score(y_test, y_pred)
    score_train = accuracy_score(y_train, model.predict(x_train_sc))

    print(f'Train accuracy: {round(score_train*100,2)}%')
    print(f'Test accuracy: {round(score*100,2)}%')
    experiment.log_metric('accuracy', score)

#%% model 1
#print(f.rf_score, f.knn_score, f.gb_score)
#%%
RF = RandomForestClassifier(n_estimators=175, max_features=25, max_depth=29, criterion='entropy', n_jobs=-1)
low_fit(RF)
experiment.end()