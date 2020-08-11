import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
from math import log, exp, sqrt
import numpy.linalg as linalg

class KalmanFilter:
    '''
    :param dim_x: dimensions of measurement vector x
    :param dim_z: dimensions of state vector z
    :param z: measurement vector
    :param x: state vector
    :param H: measurement matrix
    :param Q: measurement error covariance matrix
    :param x_0: initial estimate of state vector
    :param P_0: initial estimate of state error covariance
    :param R: state error covariance
    :param data: a dataframe that will be filtered,
        should contain `lat`, `lon`, `time` variables

    assumptions: we consider the Kalman filter with constant acceleration and non-constant velocity
    '''
    def __init__(self, H, x_0, P_0, R, data):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        
        self.H = H
        self.P_0 = P_0
        self.data = data

        self.x = x_0
        self.z = np.array([[None]*self.dim_z]).T


    
    def update():
        

##
##        self.inv = np.linalg.inv

class BayesianFilter:
    def __init__(self):
        pass

class ghkFilter:
    def __init__(self):
        pass

class MeanFilter:
    def __init__(self):
        pass

class MedianFilter:
    def __init__(self):
        pass

class HeuristicFilter:
    def __init__(self):
        pass
