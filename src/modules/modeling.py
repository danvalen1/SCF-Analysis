import pandas as pd
import pickle
import numpy as np
from modules import dataloading as dl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing as skp

def baseline(targetdir):
    sets = xywSets(targetdir)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(sets['X'], sets['y'], sets['w'], random_state=1)
    splits = {'X_train':X_train,
              'X_test':X_test,
              'y_train':y_train,
              'y_test':y_train,
              'w_train':w_train, 
              'w_test':w_test}
    
    logreg = LogisticRegression(max_iter=1000)

    # Fit to train data
    logreg.fit(X_train, y_train)

    #use the fitted model to predict on the test data
    lr_preds = logreg.predict(X_test)

    lr_f1 = metrics.f1_score(y_test, lr_preds)
    lr_prec = metrics.precision_score(y_test, lr_preds)
    lr_rec = metrics.recall_score(y_test, lr_preds)
    
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    
    return {'metrics':{'F1':lr_f1, 'Precision':lr_prec, 'Recall':lr_rec, 'AUC':AUC}, 'model':{'logreg':logreg, 'sets':{'orig':sets, 'splits':splits}}}

def weighted(targetdir):
    sets = xywSets(targetdir)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(sets['X'], sets['y'], sets['w'], random_state=1)
    
    splits = {'X_train':X_train,
              'X_test':X_test,
              'y_train':y_train,
              'y_test':y_train,
              'w_train':w_train, 
              'w_test':w_test}
    # grid search
    param_grid = {
        'penalty': ['l2'],
        'C': [.00001, .0001, .001, .01, .1, 1, 10, 100],
        'max_iter': [1000]
    }

    #create a grid search object and fit it to the data

    grid_lr=GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

    grid_lr.fit(X_train, y_train, sample_weight=w_train)

    #use the fitted model to predict on the test data
    lr_preds = grid_lr.best_estimator_.predict(X_test)

    lr_f1 = metrics.f1_score(y_test, lr_preds)
    lr_prec = metrics.precision_score(y_test, lr_preds)
    lr_rec = metrics.recall_score(y_test, lr_preds)
    
    y_pred_prob = grid_lr.best_estimator_.predict_proba(X_test)[:, 1]
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    
    return {'metrics':{'F1':lr_f1, 'Precision':lr_prec, 'Recall':lr_rec, 'AUC':AUC}, 'model':{'grid':grid_lr, 'sets':{'orig':sets, 'splits':splits}}}



def weightandscale(targetdir, iqr=(25,75), inc_drop=False, improved_params=False):
    sets = xywSets(targetdir, inc_drop)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(sets['X'], sets['y'], sets['w'], random_state=1)
    
    splits = {'X_train':X_train,
              'X_test':X_test,
              'y_train':y_train,
              'y_test':y_test,
              'w_train':w_train, 
              'w_test':w_test}
    
    scalar = skp.RobustScaler(quantile_range=iqr)
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    
    # grid search
    if improved_params:
        param_grid = {
            'penalty': ['l2'],
            'C': [.00001, .0001, .001, .01, .1, 1, 10, 100],
            'max_iter': [1000, 5000, 10000, 100000]

        }
    else:
        param_grid = {
            'penalty': ['l2'],
            'C': [.00001, .0001, .001, .01, .1, 1, 10, 100],
            'max_iter': [5000]

        }

    #create a grid search object and fit it to the data

    grid_lr=GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

    grid_lr.fit(X_train, y_train, sample_weight=w_train)

    #use the fitted model to predict on the test data
    lr_preds = grid_lr.best_estimator_.predict(X_test)

    lr_f1 = metrics.f1_score(y_test, lr_preds)
    lr_prec = metrics.precision_score(y_test, lr_preds)
    lr_rec = metrics.recall_score(y_test, lr_preds)
    
    y_pred_prob = grid_lr.best_estimator_.predict_proba(X_test)[:, 1]
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    
    return {'metrics':{'F1':lr_f1, 'Precision':lr_prec, 'Recall':lr_rec, 'AUC':AUC}, 'model':{'grid':grid_lr, 'sets':{'orig':sets, 'splits':splits}}}

def xywSets(targetdir, inc_drop=False):
    df = pd.read_stata(targetdir + 'scf2019s/p19i6.dta', columns=dl.sel_vars)
    df.columns = [x.lower() for x in df.columns]
    df.rename(columns=dl.rename_dict, inplace=True)
    df = dl.clean_SCF_df(df, neg_vals=False, modeling=True)
    
    y = df['1k_target']
    if inc_drop:
        X = df.drop(labels=['1k_target','total_income'], axis=1, inplace=False)
    else:
        X = df.drop(labels='1k_target', axis=1, inplace=False)
    
    w_df = pd.read_stata(targetdir + 'scf2019s/p19i6.dta', columns=dl.sel_vars)
    w_df.columns = [x.lower() for x in w_df.columns]
    w_df.rename(columns=dl.rename_dict, inplace=True)
    w_df = dl.clean_SCF_df(w_df, neg_vals=False, modeling=False)
    w = w_df.weighting
    
    return {'X': X, 'y': y, 'w':w}