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

if __name__ == '__main__':

    # import data
    data = pd.read_csv('census.csv')
    print('data.shape: ', data.shape)
    print('type data: ', type(data))
    #display(data.head(5))
    #print('data.head: ', data.head(5))

    data_obj = preProcess(data)
    #data_obj.getHead(n=10)
    data_obj.changeValues('income',{'<=50K':0.0, '>50K':1.0})
    #data_obj.getHead(n=10)
    data_obj.logTransform(['capital-loss', 'capital-gain'])
    #data_obj.getHead(n=10)
    # numerical features
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    data_obj.scaleMinMaxColumns(numerical)
    #data_obj.getHead(n=10)
    data_obj.getDummies()
    #data_obj.getHead(n=10)
