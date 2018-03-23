# imports here
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve as curves
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

# function definitions here
def DecisionTreeModel(X,y):
    '''
    Defines a decision tree model and fit features X to outputs y.
    In the process plot the learning curve.
    returns the regression model.
    :param X: features
    :param y: outputs
    :return: a regression model
    '''
    n = 5
    regressor = DecisionTreeRegressor(max_depth=n, random_state=0)

    train_sizes = np.linspace(1,X.shape[0]*0.8-1,9).astype(int)
    print("train_sizes: ", train_sizes)
    print("train_sizes fracs: ", train_sizes/float(X.shape[0]))

    cv = ShuffleSplit(n_splits=10, train_size=0.8, random_state=0)
    sizes, train_scores, test_scores = curves(regressor, X, y, cv=cv, train_sizes=train_sizes, scoring='r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    #plt.subplot(1,2,1)
    plt.plot(sizes, train_scores_mean,'-o')
    plt.fill_between(sizes,train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.15, color = 'b')
    #plt.subplot(1,2,2)
    plt.plot(sizes,test_scores_mean,'-o')
    plt.fill_between(sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.15, color='r')
    plt.xlabel('training size')
    plt.ylabel('r2 score')
    plt.show()
    print(regressor.get_params())


    pass


# main program
if __name__ == "__main__":
    print('running main program')
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV',axis=1)

    # some info on the data
    N_data = prices.shape[0]
    print('data keys: ', data.keys())
    print('prices shape: ', prices.shape)
    print('features shape: ', features.shape)

    # some statistics on the data
    mean_prices = np.mean(prices)
    std_prices  = np.std(prices)
    min_prices  = np.min(prices)
    max_prices  = np.max(prices)

    print('mean of prices is ${:,.2f}'.format(mean_prices))
    print('std of prices is ${:,.2f}'.format(std_prices))
    print('minimun of prices is $%f' % min_prices)
    print('maximun of prices is $%f' % (max_prices))

    DecisionTreeModel(features, prices)

