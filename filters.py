import matplotlib.pyplot as plt
import numpy as np
import math

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
    def __init__(self, z, H, Q, R, x_est, P_est):
        self.z = z
        self.H = H
        self.Q = Q
        self.R = R
        self.x_est = x_est
        self.P_est = P_est
    
    def kalmanGain(self, P, H, R):
        K = R + np.dot(np.dot(H, P), H.T)

        if type(K) != np.ndarray:
            return np.dot(P, H.T) * (1/K)
        else:
            K = np.dot(np.dot(P, H.T), np.linalg.inv(K))
            return K
            
    def update(self, P, H, R, z, x):
        '''
        :param P: state error covariance matrix
        :param H: output transition matrix
        :param R: measurement noise covariance matrix
        :param z: new measurement
        :param x: old state estimate
        :return x_new: new state estimate
        :return P: new state error covariance matrix
        '''

        K = kalmanGain(P=P, H=H, R=R)
        x_new = x + np.dot(K, z - np.dot(H, x))
        P_new = np.dot(np.eye(len(P)) - np.dot(K,H), P)

        return x_new, P_new

    def predict(self, Phi, x, P, Q):
        '''
        :param Phi: state error covariance
        :param x: current state
        :param P: current state error covariance matrix
        :param Q: process noise covariance matrix
        :return x_pred: prediction of the next state
        :return P_pred: prediction of the next state error covariance matrix
        '''
        x_pred = np.dot(Phi, x)
        P_pred = np.dot(np.dot(Phi,P), Phi.T) + Q

        return x_pred, P_pred
    
    def predictAll(self):
        x_est = self.x_init
        P_est = self.P_init
        
        for i in range(1, len(self.t)):
            Phi_i = np.array([[1, self.t[i]-self.t[i-1]], [0, 1]])
            x_pred, P_pred = predict(Phi_i, x_est, P_est, Q)
            x_est, P_est = update(P_pred, H, R, z[i], x_pred)

            X_est = np.concatenate((X_est, np.array([x_est])))
        
        return X_est

def kalmanWalk(NUM_POINTS=50, velocity=2, sigma2_R=0.1, sigma2_Q=0.0001):
    '''
    generates a constant velocity walk following the measurement model and dynamic model
    '''
    H = np.array([1,0])
    x_0 = np.array([0,velocity])
    Q = np.array([[0,0], [0,sigma2_Q]]) # process noise
    R = sigma2_R # measurement noise
    
    # initialisation
    x = np.array([x_0])
    z = np.array([np.random.normal(loc=np.dot(H, x_0), scale=math.sqrt(sigma2_R))])
    t = [0]
    
    for i in range(NUM_POINTS-1):
        delta_t = np.random.uniform(0, 10)
        t.append(t[-1] + delta_t)
        
        Phi_1 = np.array([[1, delta_t],[0, 1]])
        x_i = np.random.multivariate_normal(mean=np.dot(Phi_1, x[i-1]), cov=Q)
        x_i = np.array([x_i])
        x = np.concatenate((x, x_i))

        z_i = np.random.normal(loc=np.dot(H, x[i-1]), scale=math.sqrt(R))
        z_i = np.array([z_i])
        z = np.concatenate((z, z_i))
    
    return x, z, t
    
    

#######################################################################
# filters that I will implement when I have the time
#######################################################################

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
    
class ParticleFilter:
    def __init__(self):
        pass
