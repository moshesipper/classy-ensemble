# Classy Ensemble, copyright 2023 moshe sipper, www.moshesipper.com
# ML models

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

Mods = [LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, LinearSVC,
        DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, KNeighborsClassifier,
        XGBClassifier, LGBMClassifier]

Models = [alg for alg in Mods if hasattr(alg, 'predict_proba')]
# 8 models:
# LogisticRegression, SGDClassifier, DecisionTreeClassifier, RandomForestClassifier,
# AdaBoostClassifier, KNeighborsClassifier, XGBClassifier, LGBMClassifier


def ml_kwargs(alg):
    return {'loss': 'log_loss'} if alg == SGDClassifier else {}  # to support predict_proba
