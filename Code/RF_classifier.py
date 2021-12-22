import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('D:/ZSH/GS.csv')

X = df.drop('Target', axis=1)

y = df['Target']

np.random.seed(42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

param1 = {'n_estimators': range(10, 101, 10), 'max_features': range(1, 11, 1)}
gsearch1 = GridSearchCV(
    estimator=RandomForestClassifier(min_samples_split=10,
                                     max_depth=13,
                                     min_samples_leaf=5,
                                     random_state=10),
    param_grid=param1, scoring='roc_auc', cv=5)
gsearch1.fit(X_train, y_train)
results1 = pd.DataFrame.from_dict(gsearch1.cv_results_)


param2 = {'min_samples_leaf': range(5, 61, 5), 'min_samples_split': range(10, 201, 10)}
gsearch2 = GridSearchCV(
    estimator=RandomForestClassifier(max_features=7,
                                     n_estimators=200,
                                     max_depth=13,
                                     random_state=10),
    param_grid=param1, scoring='roc_auc', cv=5)
gsearch2.fit(X_train, y_train)
results2 = pd.DataFrame.from_dict(gsearch1.cv_results_)

param3 = {'max_depth': range(1, 13, 1), 'min_samples_split': range(10, 201, 10)}
gsearch3 = GridSearchCV(
    estimator=RandomForestClassifier(max_features=7,
                                     n_estimators=200,
                                     max_depth=13,
                                     min_samples_leaf=5,
                                     random_state=10),
    param_grid=param1, scoring='roc_auc', cv=5)
gsearch3.fit(X_train, y_train)
results3 = pd.DataFrame.from_dict(gsearch1.cv_results_)

clf1 = RandomForestClassifier(n_estimators=200,
                              min_samples_split=10,
                              max_depth=13,
                              min_samples_leaf=5,
                              max_features=7,
                              random_state=10,
                              oob_score=True)
clf1.fit(X_train, y_train)
scores1 = cross_val_score(clf1, X_train, y_train, cv=5)
clf1.score(X_train, y_train)

y_val_pred = clf1.predict(X_val)
confusion_matrix_model = confusion_matrix(y_val, y_val_pred)

print('RandomForestClassifierModel Train Score is : ', clf1.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ', clf1.score(X_val, y_val))

X_test = np.loadtxt('C:/Users/113/Desktop/X_test3.csv', dtype=np.float64, delimiter=',')
y_pred = clf1.predict(X_test)


