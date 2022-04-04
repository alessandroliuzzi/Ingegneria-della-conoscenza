from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale



def getData(path):
    data = pd.read_csv(path)
    datas = data.drop('filename', 1)
    X_data = np.array(datas.drop('label', 1))
    y_data = np.array(data['label'])
    n_class = datas.drop_duplicates(subset='label')['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle='true',stratify=y_data)
    return X_train, X_test, Y_train, Y_test, n_class

data = pd.read_csv(r"C:\Users\liuzz\OneDrive\Documenti\Ingegneria-della-Conoscenza-master\dataset/data.csv")
datas = data.drop('filename', 1)
X = np.array(datas.drop('label', 1))
y = np.array(data['label'])
X_train, X_test, Y_train, Y_test, n_class = getData(r"C:\Users\liuzz\OneDrive\Documenti\Ingegneria-della-Conoscenza-master\dataset/data.csv")


#best n_estimator for etc
param_range = np.arange(1,300,dtype=int)
train_scores, test_scores = validation_curve(ExtraTreesClassifier(),X, y, param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best max_depth for etc
param_range = np.arange(1,25, dtype=int)
train_scores, test_scores = validation_curve(ExtraTreesClassifier(),X, y, param_name="max_depth",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_depth")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best min_samples_split for etc
param_range = np.arange(2, 6, dtype=int)
train_scores, test_scores = validation_curve(ExtraTreesClassifier(),X, y, param_name="min_samples_split",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best min_samples_leaf for etc
param_range = np.arange(1,20, dtype=int)
train_scores, test_scores = validation_curve(ExtraTreesClassifier(),X, y, param_name="min_samples_leaf",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best max_features  for etc
param_range = np.arange(1,28)
train_scores, test_scores = validation_curve(ExtraTreesClassifier(),X, y, param_name="max_features",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_features")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()




#gridSearchCV for etc
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 15, 25],
    'max_features': [1, 20,28],
    'min_samples_leaf': [1,2],
    'min_samples_split': [2,4],
    'n_estimators': [100, 150, 200, 250]
}
model = ExtraTreesClassifier()
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=4, verbose=2)

gridSearch.fit(X, y)

print ("Best score:", gridSearch.best_score_)
print ("Best parameters:", gridSearch.best_params_)

