from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

def get_classifiers():
    """Get dictionary of available ML classifiers."""
    return {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'RidgeClassifier': RidgeClassifier(random_state=42),
        'SGDClassifier': SGDClassifier(random_state=42),
        'LinearSVC': LinearSVC(random_state=42)
    } 