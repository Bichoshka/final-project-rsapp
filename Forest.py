from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform
from preprocess import X,y
import numpy as np

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=451)

model = RandomForestClassifier(n_jobs=-1)
space = dict()
space['criterion'] = ['gini', 'entropy']
space['n_estimators'] = np.array(range(100, 350, 25))
space['max_features'] = ['auto', 'sqrt', 'log2']

search = RandomizedSearchCV(model, space, n_iter=100, scoring='accuracy', n_jobs=-1, cv=cv, random_state=451)
result = search.fit(X, y)
rf_score = result.best_score_
rf_params = result.best_params_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
#%%



model = GradientBoostingClassifier(verbose=1)
space = dict()
space['loss'] = ['deviance', 'exponential']
space['max_depth'] = np.array(range(2,15))
space['max_features'] = ['auto', 'sqrt', 'log2']

search = RandomizedSearchCV(model, space, n_iter=100, scoring='accuracy', n_jobs=-1, cv=cv, random_state=451)
result = search.fit(X, y)
gb_score = result.best_score_
gb_params = result.best_params_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



model = KNeighborsClassifier(n_jobs=-1)
space = dict()
space['loss'] = ['hinge', 'squared_hinge']
space['C'] = loguniform(1e-5, 100)

search = RandomizedSearchCV(model, space, n_iter=2000, scoring='accuracy', n_jobs=-1, cv=cv, random_state=451)
result = search.fit(X, y)
knn_score = result.best_score_
knn_params = result.best_params_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
np.linspace(1,50)