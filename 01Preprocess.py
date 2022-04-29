#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
df = pd.read_csv('data/train.csv', index_col='Id')
X = df.drop(['Cover_Type'], axis=1)
y = df.Cover_Type
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=451)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=451)

column_names = ['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                'Horizontal_Distance_To_Roadways']

'''t = ColumnTransformer([
    ('somename', StandardScaler(), column_names)
], remainder='passthrough')'''
def sc(i):
    scaled_features = i.copy()
    features = scaled_features[column_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[column_names] = features
    return scaled_features
x_train_sc, x_test_sc, x_val_sc = sc(x_train), sc(x_test), sc(x_val)#%%



#%%
