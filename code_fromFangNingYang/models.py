from imblearn.ensemble import EasyEnsembleClassifier
## from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from lightgbm.sklearn import LGBMClassifier

import xgboost as xgb

# def random_forest_classifier_initialise(param={}):
#     config = RandomForestClassifier().get_params()

#     config['criterion'] = 'gini'
#     config['max_depth'] = 8
#     config['min_samples_split'] = 2
#     config['min_samples_leaf'] = 1
#     config['n_estimators'] = 500
#     config['max_features'] = 20
#     config['n_jobs'] = -1
#     config['class_weight'] = 'balanced'

#     config.update(param)

#     return RandomForestClassifier(**config)


def random_forest_classifier_initialise(param={}):
    config = RandomForestClassifier().get_params()
    
    config['bootstrap'] = True
    config['class_weight'] = 'balanced'
    config['criterion'] = 'gini'
    config['max_depth'] = 90
    config['max_features'] = 'auto'    
    config['max_leaf_nodes'] = None
    config['min_impurity_decrease'] = 0.0    
    config['min_impurity_split'] = None    
    config['min_samples_leaf'] = 1    
    config['min_samples_split'] = 5
    config['min_weight_fraction_leaf'] = 0.0
    config['n_estimators'] = 1600   
    config['n_jobs'] = -1       
    config['oob_score'] = False
    config['random_state'] = None
    config['verbose'] = 0
    config['warm_start'] = False

    config.update(param)

    return RandomForestClassifier(**config)


def logstic_initialise(param={}):
    config = LogisticRegression().get_params()

    config['C'] = 1.0
    config['class_weight'] = 'balanced'
    config['dual'] = False
    config['fit_intercept'] = True
    config['intercept_scaling'] = 1
    config['max_iter'] = 2000
    config['multi_class'] = 'ovr'
    config['n_jobs'] = -1
    config['penalty'] = 'l2'
    config['solver'] = 'saga'
    config['tol'] = 1e-4
    config['verbose'] = 0

    config.update(param)

    return LogisticRegression(**config)


def ensemble_model_initialise(base_estimator=AdaBoostClassifier(), param={}):
    config = EasyEnsembleClassifier().get_params()

    config['base_estimator'] = base_estimator
    config['n_estimators'] = 50
    config['n_jobs'] = -1
    config['random_state'] = 42
    config['verbose'] = 0

    config.update(param)

    return EasyEnsembleClassifier(**config)

## parameter for "ensemble_xgb_f2_scorer.pkl" 
# def xgb_model_config_initialise(device='cpu', param={}):
#     config = xgb.XGBClassifier().get_params()

#     config['base_score'] = .5
#     config['booster'] = 'gbtree'
#     config['max_depth'] = 8
#     config['learning_rate'] = .1
#     config['n_estimators'] = 500
#     config['silent'] = True #0
#     # config['subsample'] = .8
#     config['gamma'] = 0
#     config['objective'] = 'binary:logistic'
#     config['colsample_bytree'] = .5
#     config['min_child_weight'] = 1
#     config['random_state'] = 42
#     config['predictor'] = 'cpu_predictor'
#     config['max_delta_step'] = 1



#     config.update(param)

#     return xgb.XGBClassifier(**config)


## parameter for "ensemble_xgb_1.pkl" 
def xgb_model_config_initialise(param={}):
    config = xgb.XGBClassifier().get_params()

    config['base_score'] = .5
    config['booster'] = 'gbtree'
    config['colsample_bylevel'] = 1    
    config['colsample_bytree'] = 0.6 
    config['eval_metric'] = 'auc'    
    config['gamma'] = 0.5 
    config['learning_rate'] = .1
    config['max_delta_step'] = 0      
    config['max_depth'] = 7
    config['min_child_weight'] = 1    
    config['missing'] = None  
    config['n_estimators'] = 300   
    config['n_jobs'] = -1    
    config['nthread'] = None 
    config['objective'] = 'multi:softprob'
    config['random_state'] = 0  
    config['reg_alpha'] = 0  
    config['reg_lambda'] = 1
    config['scale_pos_weight'] = 1   
    config['seed'] = None
    config['silent'] = True #0    
    config['subsample'] = 1.0   
    

    config.update(param)

    return xgb.XGBClassifier(**config)


def one_class_svm(param={}):
    config = OneClassSVM().get_params()

    config['kernel'] = 'rbf'
    config['tol'] = 1e-4
    config['nu'] = 0.5

    config.update(param)

    return OneClassSVM(**config)


def svc_initialise(param={}):
    config = SVC().get_params()
        
    config['C'] = 25
    config['cache_size'] = 200
    config['class_weight'] = None
    config['coef0'] = 0.0
    config['decision_function_shape'] = 'ovr'
    config['degree'] = 3
    config['gamma'] = 0.001
    config['kernel'] = 'rbf'
    config['max_iter'] = -1
    config['probability'] = False
    config['random_state'] = None
    config['shrinking'] = False
    config['tol'] = 1e-3
    config['verbose'] = False

    config.update(param)

    return SVC(**config)


def lgb_initialise(param={}):
    config = LGBMClassifier().get_params()                                                                             
    config['boosting_type'] = 'gbdt'
    config['class_weight'] = None
    config['colsample_bytree'] = 0.7
    config['importance_type'] = 'split'
    config['is_unbalance'] = True
    config['learning_rate'] = 0.05
    config['max_depth'] = 4   
    config['min_child_samples'] = 20
    config['min_child_weight'] = 0.001
    config['min_split_gain'] = 0.0
    config['n_estimators'] = 600
    config['n_jobs'] = -1
    config['nthread'] = 3
    config['num_leaves'] = 8
    config['objective'] = 'binary'
    config['random_state'] = None
    config['reg_alpha'] = 0
    config['reg_lambda'] = 0    
    config['seed'] = 777
    config['silent'] = False
    config['subsample'] = 0.8
    config['subsample_for_bin'] = 200000
    config['subsample_freq'] = 0
    
    config.update(param)

    return LGBMClassifier(**config)