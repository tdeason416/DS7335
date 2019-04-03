#!/usr/bin/python
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc as auc_func, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from collections import OrderedDict
from itertools import cycle


# adapt this code below to run your analysis
 
# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each
 
# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
 
# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function


################
### Child Classes, to be used by Grid Search class
################

class GridDefault(object):
    '''
    Default class for all other grid search classifiers (not used)
    --------
    ATTRIBUTES
    modeltype: SkLearn Type model
    parameters: list of strings
        - List of parameters which will be tuned by grid search
    tuning_options: dict { string : list }
        - Possible values to be used by ^ parameters
    tuning_state: dict {string : ( numeric | string )}
        - Current state of model parameters
    current_param: int
        - Represents current parameter being processed
    param_idx: int
        - Represents list index of parameter being processed
    max_state: dict {string : ( numeric | string )}
        - Final in scope value for model prameters
    --------
    METHODS
    fit_and_predict:
        - Test model accuracy for a single epoch
            * Inputs: training and testing data
            * Returns: predicted probabilities based on test data
    cycle_params:
        - Change Model params in place
            * Inputs: None
            * Returns: bool, True if params were cycled, False if iterations are complete
    '''
    def __init__(self):
        self.modelname = "None"
        self.modeltype = LogisticRegression # dummy value
        self.defaults = {
            'random_state': 42,
            'default_2' : 'who_cares'
        }
        self.parameters = ["Param1", 'Param2']
        self.tuning_options = {
            'Param1': ['Option1', 'Option2'],
            'Param2': np.linspace(0,1, 50)
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}
  
    def fit_and_predict(self, X_tr, y_tr, X_te, y_te):
        model_params = { **self.defaults,  **{p: self.tuning_state[p] for p in self.parameters}}
        model = self.modeltype(**model_params)    
        model.fit(X_tr, y_tr)
        return (model.predict_proba(X_te)[:,1], model.predict(X_te), y_te)

class GridLogisticReg(GridDefault):
    '''
    Logistic regression grid search object, inherits GridDefault
    '''
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = "LogisticRegression"
        self.modeltype = LogisticRegression
        self.defaults = {
            'random_state': 42,
            'n_jobs' : -1
        }
        self.parameters = ['C', 'penalty']
        self.tuning_options = {
            'C': np.linspace(.001,1, 50),
            'penalty': ['l1', 'l2']
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}


## SVM was removed because it doe not return probabilities (and it's really slow running)
# class GridSVM


class GridDecisionTree(GridDefault):
    '''
    Decision Tree grid search object, inherits GridDefault
    '''
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = "DecisionTreeClassifier"
        self.modeltype = DecisionTreeClassifier
        self.defaults = {
            'random_state': 42
        }
        self.parameters = [
            "min_weight_fraction_leaf",
            'max_depth'
            # 'min_impurity_split'
            ]
        self.tuning_options = {
            'min_weight_fraction_leaf': np.linspace(0, .2, 20),
            'max_depth': np.arange(20, 3, -1),
            # 'min_impurity_split': np.linspace(0, .5, 20)
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridNaiveBayes(GridDefault):
    '''
    Naive Bayes Classifier grid search object, inherits GridDefault
    **THIS WAS REMOVED FROM THE GRID SEARCH PROCESS**
    '''
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = "GaussianNaiveBayes"
        self.modeltype = MultinomialNB
        self.defaults = {}
        self.parameters = [
            "alpha",
            ]
        self.tuning_options = {
            'alpha': np.linspace(0, .3, 30)
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridRandomForest(GridDefault):
    '''
    Random Forest Classifier grid search object, inherits GridDefault
    '''
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = 'RandomForestClassifier'
        self.modeltype = RandomForestClassifier
        self.defaults = {
            'random_state' : 42,
            'min_samples_leaf': 1,
            'n_jobs' : -1
        }
        self.parameters = [
            'n_estimators',
            'max_depth'
            ]
        self.tuning_options = {
            'n_estimators': (2 ** np.linspace(1, 10, 25)).astype(int),
            'max_depth': np.arange(1, 19, 3)
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}

class GridXGBoost(GridDefault):
    '''
    XGBoost Classifier grid search object inherits GridDefault
    '''
    def __init__(self):
        GridDefault.__init__(self)
        self.modelname = 'XGBoostClassifier'
        self.modeltype = XGBClassifier
        self.defaults = {
            'random_state' : 42,
            'n_jobs' : -1
        }
        self.parameters = [
            'n_estimators',
            'max_depth'
            ]
        self.tuning_options = {
            'n_estimators': (2 ** np.linspace(0, 10, 25)).astype(int),
            'max_depth': np.arange(1, 50, 3)
        }
        self.tuning_state = {k: v[0] for k, v in self.tuning_options.items()}
        self.numeric_state = [0 for _ in self.tuning_state.values()]
        # self.current_param = 0
        # self.param_idx = 0
        self.max_state = { s: self.tuning_options[s][-1] for s in self.parameters}


################
### Main Class
################

class GridSearch(object):
    '''
    Completes grid search for sevearal models using a cross validated set
    --------
    ATTRIBUTES
    data: np.array {dims (n x k)}
      - Numpy array containing all independent data (no labels)
    labels: np.array {dims (n x 1) or (n})
      - Numpy array containing labels, or y-values
    data_cols: list of strings (default none)
      - Column names for independent variables, if none will be ['1', '2', ...'n']
    label_name: int (default=-'y')
      - name of label column, if none will be 'y'
    numfolds: int (default 5)
      - number of folds in K-folds cross validation set
    output_folder: str (default os.getcwd() [current folder])
        -   folder location to save charts and spreadsheets of output data
    train_test_folds: tuple (length (numfolds) of bool arrays)
        -   represents the train test folds to be used in all models
    results_auc: dict (key: model_name, value: np.array)
        -   area under the curve values (ratio)
    results_precision: dict (key: model_name, value: np.array)
        -   precision for a given model (ratio)
    results_recall: dict (key: model_name, value: np.array)
        -   precision for a given parameter set (ratio)
    results_accuracy: dict (key: model_name, value: np.array)
        -   accuracy for a given paraeter set(ratio)
    --------
    METHODS
    optimize_all_models
    optimize_model
    output_auc_plot
    _save_model_data
    diff_measure
    run_one_model
    save_results
    '''
    def __init__(self, data, labels, data_cols=None, label_name='y', numfolds=5, output_folder=os.getcwd()):
        if not data_cols:
            data_cols = np.arange(data.shape[1])
        self.data_cols = data_cols
        self.data = data
        self.labels = labels
        self.label_names = label_name
        self.numfolds = numfolds
        self.output_folder = output_folder
        self.colors = cycle(['blue','green','red','cyan','magenta','black'])
        folds = np.random.random(self.data.shape[0])
        self.train_folds_X = [data[(folds <= i) | (folds > i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.train_folds_y = [labels[(folds <= i) | (folds > i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.test_folds_X = [data[(folds > i) & (folds <= i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.test_folds_y = [labels[(folds > i) & (folds <= i+1/numfolds)]  for i in np.linspace(0,1-1/numfolds, numfolds)]
        self.train_test_folds = tuple(zip(
            self.train_folds_X, 
            self.train_folds_y, 
            self.test_folds_X, 
            self.test_folds_y
        ))
        self.models = { 
            'logis': GridLogisticReg(), 
            'dtree': GridDecisionTree(),
            # 'nbayes': GridNaiveBayes(),
            'rforest': GridRandomForest(),
            'xgboost': GridXGBoost()
        }
        self.results_auc = {
          model_name :  
              np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters]))\
                  .reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
              for model_name, model_i in self.models.items()
          }
        self.results_precision = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters]))\
                  .reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        self.results_recall = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters]))\
                  .reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        self.results_accuracy = {
            model_name : \
                np.zeros(
                  np.prod([len(model_i.tuning_options[n]) for n in model_i.parameters]))\
                  .reshape(*[len(model_i.tuning_options[n]) for n in model_i.parameters])
                for model_name, model_i in self.models.items()
            }
        if 'data' not in os.listdir(output_folder):
            os.mkdir('data')
        if 'charts' not in os.listdir(output_folder):
            os.mkdir('charts')

    def optimize_all_models(self):
        '''
        Run grid search on all models.
        '''
        for m_name in self.models.keys():
            print(m_name)
            self.optimize_model(m_name)

    def output_auc_plot(self, model_idx, results):
        '''
        Outputs auc plot for model
        --------
        model_idx: str
        y_te : np.array
        probs: np.array
        '''
        plt.figure(figsize=(10,10))
        plt.title('Receiver Operating Characteristic')
        sum_aucs = 0
        for fold in range(self.numfolds):
            # y_te = self.test_folds_y[fold]
            # print(len(results[fold]))
            # print(results[fold])
            probs, _, y_te = results[fold]
            fpr, tpr, _ = roc_curve(y_te, probs)
            roc_auc = auc_func(fpr, tpr)
            plt.plot(fpr, tpr, color=next(self.colors), label = 'AUC = %0.3f' % roc_auc)
            sum_aucs += roc_auc
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        modi = self.models[model_idx]
        file_name = self.output_folder + "/charts/" + \
            modi.modelname + "_" +\
            "::".join([str(p) + "-" +  str(v) for p,v in modi.tuning_state.items()]) +\
            "__auc-{}.png".format(sum_aucs / self.numfolds)
        plt.savefig(file_name)
        plt.close()

    def save_model_data(self, model_idx, results):
        '''
        Save model output into array based on current information
        --------
        INPUTS
        model_idx: str ('logis' | 'dtree' | 'nbayes' | 'rforest' | 'xgboost')
        results: list of lists
        --------
        RETURNS
        auc: float
            - Area under ROC curve
        '''
        aucs = []
        precs = []
        recs = []
        accs = []
        for y_probs, y_pred, y_te in results:
            mtrx = confusion_matrix(y_te, y_pred)
            tp = mtrx[0,0]
            fp = mtrx[0,1]
            fn = mtrx[1,0]
            fpr, tpr, _ = roc_curve(y_te, y_probs)
            aucs.append(auc_func(fpr, tpr))
            precs.append( float(tp) / (tp + fp) )
            recs.append( float(tp) / (tp + fn) )
            accs.append( float((y_pred == y_te).sum()) / y_te.shape[0] )
        auc = sum(aucs) / len(aucs)
        prec = sum(accs) / len(accs)
        rec = sum(recs) / len(recs)
        accs = sum(accs) / len(accs)
        modi = self.models[model_idx]
        print(modi.tuning_state)
        ## Area under ROC curve
        self.results_auc[model_idx]\
            [modi.numeric_state[0], modi.numeric_state[1]] = auc
            # [modi.current_param][modi.param_idx] = auc
        ## Precision
        self.results_precision[model_idx]\
            [modi.numeric_state[0], modi.numeric_state[1]] = prec
            # [modi.current_param][modi.param_idx] = prec
        ## Recall
        self.results_recall[model_idx]\
            [modi.numeric_state[0], modi.numeric_state[1]] = rec
            # [modi.current_param][modi.param_idx] = rec
        ## Accuracy
        self.results_accuracy[model_idx]\
              [modi.numeric_state[0], modi.numeric_state[1]] = accs
            # [modi.current_param][modi.param_idx] = acc
        return auc

    def diff_measure(self, model_idx, param, p_int, min_opts=8):
        '''
        Look for local maxima in 8 points along the param space
        --------
        INPUTS
          model_idx: str ('logis' | 'dtree' | 'nbayes' | 'rforest' | 'xgboost')
          param: str
              - Name of parameter which is being modified
          p_int: int (0 | 1)
              - corresponding integer value to parameter
          min_opts: int (default 8)
              - Minimum number of parameter choices to be considered for diff_measure
        --------
        RETURNS
          None
        '''
        modi = self.models[model_idx]
        aucs = OrderedDict()
        settings = modi.tuning_options[param]
        self.models[model_idx].tuning_state[param] = settings[0]
        print(self.models[model_idx].numeric_state)
        aucs[settings[0]] = self.run_one_model(model_idx)
        for idx in np.arange(len(settings) / min_opts, len(settings), len(settings)/ min_opts):
            self.models[model_idx].tuning_state[param] = settings[int(idx)]
            self.models[model_idx].numeric_state[p_int] = int(idx)
            aucs[settings[int(idx)]] = self.run_one_model(model_idx)
        self.models[model_idx].tuning_state[param] = settings[-1]
        aucs[settings[-1]] = self.run_one_model(model_idx)
        vals = tuple(aucs.values())
        keys = tuple(aucs.keys())
        for auc_idx in range(1, len(vals))[:-1]:
            if (vals[auc_idx] > vals[auc_idx -1]) and (vals[auc_idx] > vals[auc_idx + 1]):
                for s_int, sett in enumerate(settings[[i for i,v in enumerate(settings) if v == keys[auc_idx]][0]:
                                       [i for i,v in enumerate(settings) if v == keys[auc_idx + 1]][0]]):
                    self.models[model_idx].tuning_state[param] = sett
                    self.models[model_idx].numeric_state[p_int] = s_int
                    aucs[int(idx)] = self.run_one_model(model_idx)

    def split_param(self, model_idx, param, min_params=10):
        '''
        '''
        jump = min_params - 2
        settings = self.models[model_idx].tuning_options[param]
        if len(settings) >= min_params:
            idxs = np.linspace(0, len(settings)-1, min_params).astype(int)
            vals = [settings[idx] for idx in idxs]
            return zip(idxs, vals)
        else:
            return enumerate(settings)

    def fill_missing(self, model_idx, row_col, num, prev, nxt, row=True):
        '''
        '''
        p_0 = self.models[model_idx].parameters[0]
        p_1 = self.models[model_idx].parameters[1]
        s_0 = self.models[model_idx].tuning_options[p_0]
        s_1 = self.models[model_idx].tuning_options[p_1]

        auc_mtrx = self.results_auc[model_idx]
        dummy_mtrx = np.arange(auc_mtrx.size).reshape(*auc_mtrx.shape)
        ### find start point
        idx_min = dummy_mtrx[auc_mtrx == prev][0]
        idx_min_x = idx_min % auc_mtrx.shape[0]
        idx_min_y = int(idx_min / auc_mtrx.shape[1])
        ### find end point
        idx_max = dummy_mtrx[auc_mtrx == nxt][0]
        idx_max_x = idx_max % auc_mtrx.shape[0]
        idx_max_y = int(idx_max / auc_mtrx.shape[1])
        ### fill in missing values
        if row_col == 'row':
            self.models[model_idx].tuning_state[p_0] = s_0[num]
            for col in range(idx_min_x + 1, idx_max_x - 1):
                self.models[model_idx].numeric_state = [num, col]
                self.models[model_idx].tuning_state[p_1] = s_1[col]
                self.run_one_model(model_idx)
        elif row_col == 'col':
            self.models[model_idx].tuning_state[p_1] = s_1[num]
            for row in range(idx_min_y + 1, idx_max_y - 1):
                self.models[model_idx].numeric_state = [row, num]
                self.models[model_idx].tuning_state[p_0] = s_0[row]
                self.run_one_model(model_idx)

    def optimize_model(self, model_idx, min_params=10):
        '''
        Complete grid search along parameter space
        --------
        INPUTS
        model_idx: str ('logis' | 'dtree' | 'nbayes' | 'rforest' | 'xgboost')
            - Which model to optimize
        min_params: int (default 10)
            - Minimum number of parameter options to use optimizer
        --------
        RETURNS
        None:
            - object modified in place
        '''
        modi = self.models[model_idx]
        s_y = list(self.split_param(model_idx, modi.parameters[0]))
        p_y = modi.parameters[0]
        s_x = list(self.split_param(model_idx, modi.parameters[1]))
        p_x = modi.parameters[1]
        for y_idx, y_val in s_y:
          self.models[model_idx].tuning_state[p_y] = y_val
          self.models[model_idx].numeric_state[0] = y_idx
          for x_idx, x_val in s_x:
              self.models[model_idx].tuning_state[p_x] = x_val
              self.models[model_idx].numeric_state[1] = x_idx
              self.run_one_model(model_idx)
        
        auc_mtrx = self.results_auc[model_idx]
        mtrx = auc_mtrx[auc_mtrx > 0].reshape(len(s_y), len(s_x))
        for r in range(1,mtrx.shape[0] - 1):
            for c in range(1, mtrx.shape[1] - 1):
                if mtrx[r-1, c] > mtrx[r, c] and mtrx[r, c] < mtrx[r + 1, c]:
                    self.fill_missing(model_idx, 'col', c, mtrx[r-1, c], mtrx[r + 1, c])
                if mtrx[r, c-1] > mtrx[r, c] and mtrx[r, c] < mtrx[r, c + 1]:
                    self.fill_missing(model_idx, 'row', r, mtrx[r, c-1], mtrx[r, c + 1])            

    def run_one_model(self, model_idx):
        '''
        Run model for a single configuration type on a single model
        --------
        INPUTS
        model_idx: str ('logis' | 'dtree' | 'nbayes' | 'rforest' | 'xgboost')
            -  which type of model to fit
        --------
        RETURNS
        None: 
            -  Model output is saved in place
        '''
        results = []
        for X_tr, y_tr, X_te, y_te in self.train_test_folds:
            results.append(self.models[model_idx].fit_and_predict(X_tr, y_tr, X_te, y_te))
        ## output model performance to numpy array
        auc = self.save_model_data(model_idx, results)
        self.output_auc_plot(model_idx, results)
        return auc

    def save_results(self):
        '''
        Saves ROC plot for best version of each model and outputs results to .csv
        --------
        INPUTS
        None
        --------
        RETURNS
        None
        '''
        for model_idx, model_i in self.models.items():
            row_labels = ','.join(
                [str(x) for x in model_i.tuning_options[model_i.parameters[0]]]
            ) + "\n"
            col_labels = ',' + ','.join(
                [str(x) for x in model_i.tuning_options[model_i.parameters[1]]]
            ) + "\n"

            auc_out = col_labels
            prec_out = col_labels
            rec_out = col_labels
            acc_out = col_labels

            for r_idx, r_lbl in enumerate(row_labels.split(',')):
                auc_out += \
                    str(r_lbl) + "," + \
                    ','.join([str(x) for x in self.results_auc[model_idx][r_idx, :]]) +\
                    "\n"
                prec_out += str(r_lbl) + "," + \
                    ','.join([str(x) for x in self.results_precision[model_idx][r_idx, :]]) +\
                    "\n"
                rec_out += str(r_lbl) + "," + \
                    ','.join([str(x) for x in self.results_recall[model_idx][r_idx, :]]) +\
                    "\n"
                acc_out += str(r_lbl) + "," + \
                    ','.join([str(x) for x in self.results_accuracy[model_idx][r_idx, :]]) +\
                    "\n"

            x_y = self.output_folder + "/data/" +\
                self.models[model_idx].modelname +\
                "-{}_" + \
                "x:{}::y:{}.csv".format(*self.models[model_idx].parameters)

            print(x_y.format('auc'))

            open(x_y.format('auc'), 'w').write(auc_out.strip())
            open(x_y.format('precision'), 'w').write(prec_out.strip())
            open(x_y.format('recall'), 'w').write(rec_out.strip())
            open(x_y.format('accuracy'), 'w').write(acc_out.strip())

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    gs = GridSearch(X, y)
    gs.optimize_all_models()
    gs.save_results()

    # def test_model(self):
      # void


 






# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
# L = np.ones(M.shape[0])
# n_folds = 5
 
# data = (M, L, n_folds)
 
# def run(a_clf, data, clf_hyper={}):
#   M, L, n_folds = data # unpack data containter
#   kf = KFold(n_splits=n_folds) # Establish the cross validation
#   ret = {} # classic explicaiton of results
 
#   for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
#     clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
 
#     clf.fit(M[train_index], L[train_index])
 
#     pred = clf.predict(M[test_index])
 
#     ret[ids]= {'clf': clf,
#                'train_index': train_index,
#                'test_index': test_index,
#                'accuracy': accuracy_score(L[test_index], pred)}
#   return ret
 
# results = run(RandomForestClassifier, data, clf_hyper={})
