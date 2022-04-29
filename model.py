from preprocess import *
import Forest as f
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=451)
final = []
def low_fit(model):
    model.fit(x_train_sc, y_train) #фит

    y_pred=model.predict(x_test_sc)#предсказание

    score=f1_score(y_test, y_pred)
    score_train = f1_score(y_train, model.predict(x_train_sc))

    print(f'Train F1: {round(score_train*100,2)}%')
    print(f'Test F1: {round(score*100,2)}%')

    final.append([str(model), round(score*100,2)])

#%% model 1
print(f.rf_score, f.pac_score, f.gb_score)
#%%
