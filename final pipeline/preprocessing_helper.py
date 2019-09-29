
import matplotlib.pyplot as plt # plotting
import pandas as pd # data manipulation and analysis
import numpy as np # numerical computation
import pickle

import scipy
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import random
import math

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime


# helper methods
"""The methods below take as input a list of features - this is 
the data, normalised and chopped. The output is some feature of 
the time series, for instance it could be the mean, the standard 
deviation, the max, the min, the norm gradient mean etc. 
"""

def the_means(feature_list):
    feat_means = []
    for i in feature_list:
        transposed = np.transpose(i)
        means = []
        for j in transposed:
            means.append(np.mean(j))
        feat_means.append(means)
    return feat_means

def the_stds(feature_list):
    feat_stds = []
    for i in feature_list:
        transposed = np.transpose(i)
        stds = []
        for j in transposed:
            stds.append(np.std(j))
        feat_stds.append(stds)
    return feat_stds

def the_max(feature_list):
    feat_max = []
    for i in feature_list:
        transposed = np.transpose(i)
        maxes = []
        for j in transposed:
            maxes.append(max(j))
        feat_max.append(maxes)
    return feat_max

def the_min(feature_list):
    feat_min = []
    for i in feature_list:
        transposed = np.transpose(i)
        mins1 = []
        for j in transposed:
            mins1.append(min(j))
        feat_min.append(mins1)
    return feat_min
        
def get_gradient(feature_list):# returns the gradient of everything
    feat_grad = []
    for i in feature_list:
        transposed = np.transpose(i)
        # find the gradient at each point
        gradients = []
        for j in range(len(transposed)):
            grad_feat = []
            for k in range(len(transposed[j])):
                if k>=1:
                    grad_feat.append(transposed[j][k]-transposed[j][k-1])
            gradients.append(grad_feat)
        feat_grad.append(np.transpose(gradients))
    return feat_grad
        
def non_zero_grad_mean45(feature_list):# 2d
    feature_grad = get_gradient(feature_list)
    non_zero_grad_means = []
    for i in feature_grad:
        transposed = np.transpose(i)
        non_zero_grad = [[],[]]
        for j in [4,5]:
            for k in range(len(transposed[j])):
                if transposed[j][k]!=0:
                    non_zero_grad[j-4].append(transposed[j][k])
        a0 = np.mean([abs(i) for i in non_zero_grad[0]])
        a1 = np.mean([abs(i) for i in non_zero_grad[0]])
        non_zero_grad_means.append([a0,a1])
    return non_zero_grad_means

def gradient_norm_mean(feature_list):
    feat_grad = get_gradient(feature_list)
        
    # we want a gradient (absolute value) mean as a feature
    feat_grad = map(abs,feat_grad)
    gradient_mean = the_means(feat_grad)
    
    return gradient_mean

def double_derivative_norm_mean(feature_list):# 10 D
    double_derivative = get_gradient(get_gradient(feature_list))
    
    # we want the absolute value mean of the double derivative
    double_derivative = map(abs,double_derivative)
    double_derivative_mean = the_means(double_derivative)
    
    return double_derivative_mean

# noticed that features 4, 5 and 6 are often 50 for many data points in a row, in other
# words the gradient is 0 quite alot. We therefore add two new features which are 
# 1) the number of data points where the gradient is 0 divided by the number of times the gradient is zero
# we do this for numbers 4, 5 and 6
def grad_new_features(feature_list):
    feature_grad = get_gradient(feature_list)
    new_features_list = []
    for i in feature_grad:
        transposed = np.transpose(i)
        new_feature = []
        for j in [4,5,6]:
            count=0
            for k in range(len(transposed[j])):
                if transposed[j][k]==0: count+=1
            try: new_feature.append(count/len(transposed[j]))
            except ZeroDivisionError: new_feature.append(0)
        new_features_list.append(new_feature)
    return new_features_list


# returns feature list of how many times the gradient changes sign
def turns(feature_list):
    feature_grad = get_gradient(feature_list)
    changes_sign_list = []
    for i in feature_grad:
        transposed = np.transpose(i)
        changes_sign = []
        for j in range(len(transposed)):
            count=0
            for k in range(1,len(transposed[j])):
                if (transposed[j][k]>0 and transposed[j][k-1]<=0) or (transposed[j][k]<=0 and transposed[j][k]>0):
                    count+=1
            try: changes_sign.append(count/len(transposed[j]))
            except ZeroDivisionError: changes_sign.append(0)
        changes_sign_list.append(changes_sign)
    return changes_sign_list


def feat_arima_coefs(feature_list):
    feat_arima = []
    # first 1
    for i in feature_list:
        transposed = np.transpose(i)
        series = transposed[1]
        model = ARIMA(series, order=(1,1,0))
        model_fit = model.fit(disp=0)
        
        params1 = model_fit.params[:]
        
        series = transposed[4]
        model = ARIMA(series, order=(1,1,0))
        model_fit = model.fit(disp=0)
        
        params4 = model_fit.params[:]
        
        serise = transposed[6]
        model = ARIMA(series, order=(1,1,0))
        model_fit = model.fit(disp=0)
        
        params6 = model_fit.params[:]
        
        feat_arima.append(list(params1)+list(params4)+list(params6))
        
    return feat_arima


def get_features_map(path_string="../data/data_lstm/X_raw.npy",pickle_name="time_series_features.pickle",pickle=True,arima_works=True):

    time_series = load_time_series(path_string)

    # means
    feat_means = the_means(time_series)# 10 D
    # stds
    feat_stds = the_stds(time_series)# 10 D
    # maxes
    feat_max = the_max(time_series)# 10 D
    # min
    feat_min = the_min(time_series)# 10 D
    # gradient mean
    feat_grad = gradient_norm_mean(time_series)# 10 D
    # new features from 4,5,6
    new_feat1 = grad_new_features(time_series)# 3 D
    # non zero grad 4,5
    non_zero_grad = non_zero_grad_mean45(time_series)# 2 D
    # the double derivative mean
    feat_dd = double_derivative_norm_mean(time_series)# 10 D
    # the number of times the gradient changes sign
    feat_changes_sign = turns(time_series)# 10 D
    
    # the arima stuff, try catch in the event that it rejects the data, that there is no fit
    try: 
        feat_arima = feat_arima_coefs(time_series)
        big_features_list = [feat_means,feat_stds,feat_max,feat_min,feat_grad,new_feat1,
        non_zero_grad,feat_dd,feat_changes_sign,feat_arima]
    except: 
        arima_works =False
        big_features_list = [feat_means,feat_stds,feat_max,feat_min,feat_grad,new_feat1,
        non_zero_grad,feat_dd,feat_changes_sign]
    
    
    for i in big_features_list: # trace
        try: print(len(i))
        except: print(i)
    #input()
    ###
    
    # first check that they are all the same length
    l = len(feat_means)
    for i in big_features_list:
        if len(i)!=l: raise Exception("features lists are not the same size, they should be")
        
    complete_features_list = big_features_list[0]
    for j in range(1,len(big_features_list)):
        for k in range(len(big_features_list[0])):
            complete_features_list[k] = complete_features_list[k]+big_features_list[j][k]
    
    #for j in big_features_list:
     #   if j==feat_means: complete_features_list=[j]
      #  else:
       #     for k in range(l):
        #        complete_features_list[k] = complete_features_list[k]+j[k]
    
    # pickle the data
    if pickle:
        np.array(complete_features_list).dump(open('data_lstm/x_feature_arima.npy', 'wb'))
    
    return complete_features_list
    

def load_time_series(path_string='../data/data_lstm/X_raw.npy'):
    time_series_x = np.load(path_string)
    return time_series_x
