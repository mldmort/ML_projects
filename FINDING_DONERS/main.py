import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class preProcess(object):
    def __init__(self, data):
        '''
        :param data: should be a pandas DataFrame
        '''
        self.data = data

    def getHead(self,n):
        display(self.data.head(n))

    def getData(self):
        return self.data

    def myfunc(self):
        pass

    def scaleMinMaxColumns(self, col_names):
        '''
        :param col_names: column names, list of strings
        '''
        scalar = MinMaxScaler()
        self.data[col_names] = scalar.fit_transform(self.data[col_names])

    def scaleStandardColumns(self, col_names):
        '''
        :param col_names: column names, list of strings
        '''
        scalar = StandardScaler()
        self.data[col_names] = scalar.fit_transform(self.data[col_names])

    def getDummies(self):
        self.data = pd.get_dummies(self.data)

    def changeValues(self, col_name, change_rule_dic):
        '''
        :param col_name: column name in the DataFrame
        :param change_rule_dic: a dictionary with keys equal to old values and values equal to new values
        '''
        for old_value, new_value in change_rule_dic.items():
            self.data[col_name] = self.data[col_name].apply(lambda x: new_value if x == old_value else x)

    def logTransform(self, col_names):
        '''
        :param col_name: column names, list string
        '''
        self.data[col_names] = self.data[col_names].apply(lambda x: np.log(x+1))

from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
def quick_train_predict(learner, X_train, y_train, X_test, y_test):
    '''
    :param learner: a learner object which should have attributes fit and predict
    :param X_train: training features
    :param y_train: training ground truth
    :param X_test: testing features
    :param y_test: testing ground truth
    :return: a dictionary containing the results
    '''

    results = {}
    learner = learner.fit(X_train,y_train)

    y_predict = learner.predict(X_test)

    accuracy = accuracy_score(y_test,y_predict)
    fb_score = fbeta_score(y_test,y_predict,beta=0.5)
    results['accuracy'] = accuracy
    results['fb_score'] = fb_score

    return results

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
def grid_search(learner,params, X_train, y_train):
    '''
    :param learner: the learner
    :param params: a dictionary with keys equal to parameters in string format and the values are lists of the values
    :return: a dictionary containing the results
    '''
    results = {}
    scorer = make_scorer(fbeta_score, beta=0.5)
    cls = GridSearchCV(learner,params,scorer)
    cls.fit(X_train, y_train)
    results['best_est'] = cls.best_estimator_
    results['best_score'] = cls.best_score_
    results['best_params'] = cls.best_params_
    return results

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
if __name__ == '__main__':

    # import data
    data = pd.read_csv('census.csv')
    print('data.shape: ', data.shape)
    print('type data: ', type(data))
    #display(data.head(5))
    #print('data.head: ', data.head(5))

    # define a data object
    data_obj = preProcess(data)
    #data_obj.getHead(n=10)

    # change the outcome values to a 0-1 representation
    data_obj.changeValues('income',{'<=50K':0.0, '>50K':1.0})
    #data_obj.getHead(n=10)

    # log transform the spread-out data to contain their variations
    data_obj.logTransform(['capital-loss', 'capital-gain'])
    #data_obj.getHead(n=10)

    # numerical features
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # scale numerical features to vary between 0 and 1
    data_obj.scaleMinMaxColumns(numerical)
    #data_obj.getHead(n=10)

    # use pandas library to turn categorical features to one-hot vectors
    data_obj.getDummies()
    #data_obj.getHead(n=10)

    # get the data
    data_pp = data_obj.getData()
    features = data_pp.drop('income',axis = 1)
    income = data_pp['income']

    # split and shuffle the data into test and training parts
    X_train, X_test, y_train, y_test = train_test_split(features,income, test_size=0.2, random_state=0)


    estimator_a = 'AdaBoost'
    estimator_b = 'SVC'
    estimator_c = 'LogisticRegression'
    # pick your favorate classifier
    estimator = estimator_c

    learner_a = AdaBoostClassifier()
    learner_b = SVC()
    learner_c = LogisticRegression()
    learner_dict = {estimator_a:learner_a, estimator_b:learner_b, estimator_c:learner_c}
    learner = None
    if estimator == 'AdaBoost':
        learner = learner_a
    elif estimator == 'SVC':
        learner = learner_b
    elif estimator == 'LogisticRegression':
        learner = learner_c

    # test your estimator with default parameters
    for name, lrn in learner_dict.items():
        results = quick_train_predict(lrn,X_train, y_train, X_test, y_test)
        print('estimator: ', name, 'Accuracy: ', results['accuracy'], 'fbeta score: ', results['fb_score'])

    # check params of estimator and define a range of parameters
    if estimator == estimator_a:
        print('estimator: ', estimator, 'params: ', learner.get_params())
        param_dict = {'learning_rate': [0.9, 1.0, 1.1], 'n_estimators': [50, 40, 30]}
        pass
    elif estimator == estimator_b:
        print('estimator: ', estimator, 'params: ', learner.get_params())
        param_dict = {'C': [0.8, 0.9, 1., 1.1]}
        pass
    elif estimator == estimator_c:
        print('estimator: ', estimator, 'params: ', learner.get_params())
        param_dict = {'C': [0.8, 0.9, 1., 1.1]}
        pass

    # optimize estimator with grid search
    results = grid_search(learner, param_dict, X_train, y_train)
    print('estimator: ',estimator,'best classifier params: ',results['best_params'])
    print('estimator: ',estimator,'best classifier score: ',results['best_score'])

