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


    
    def genNextStep():
        pass

def genTrajectory(NUM_POINTS = 20):
    # randomly generate distance to take for each time step
    x_points = np.array([random.randrange(start=-50, stop=100) for i in range(NUM_POINTS)])/100
    y_points = np.array([random.randrange(start=-50, stop=100) for i in range(NUM_POINTS)])/100
    t_points = np.array([random.randrange(start=0, stop=1000) for i in range(NUM_POINTS)])/100
    rad_inacc = np.array([random.randrange(start=0, stop=100) for i in range(NUM_POINTS)])/100
    
    # trajectory is just a 3D array of latitude, longitude, time
    traj = np.zeros(shape=(NUM_POINTS+1,4))

    for i in range(1, NUM_POINTS+1):
        traj[i,0] = traj[i-1,0] + x_points[i-1]
        traj[i,1] = traj[i-1,1] + y_points[i-1]
        traj[i,2] = traj[i-1,2] + t_points[i-1]
        
    traj[0,3] = random.randrange(start=0,stop=100)/100
    traj[1:,3] = rad_inacc
    
    traj = pd.DataFrame(traj)
    traj.columns = ["lon", "lat", "time", "inacc_radius"]
    
    return traj
        
if "__name__" == "__main__":
    # test if the velocity model works
    sigma_sq = 2.5
    
    H = np.array([[1,0,0,0],[0,1,0,0]])
    R = sigma_sq * eye(2)
    
    data = genTrajectory()
    x_0 = np.array([data.lon[0], data.lat[0]])
    
    kf = KalmanFilter(H, x_0, P_0, R, data)

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
