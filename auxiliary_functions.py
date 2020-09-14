import numpy as np
import pandas as pd
import random
from statistics import mean, median
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import numba

def genTrajectory(NUM_POINTS = 20 , x_range=[100,110], y_range=[0,5], timestep_range=[60, 600], rad_range=[0,100]):
    '''
    randomly generate distance to take for each time step
    '''
    startDate = date(2019, 4, 13)
    
    x_points = np.random.uniform(low=x_range[0], high=x_range[1], size=NUM_POINTS)
    y_points = np.random.uniform(low=y_range[0], high=y_range[1], size=NUM_POINTS)
    t_points = np.random.uniform(low=timestep_range[0], high=timestep_range[1], size=NUM_POINTS)
    rad_inacc = np.random.uniform(low=rad_range[0], high=rad_range[1], size=NUM_POINTS)
    
    # trajectory is just a 4D array of latitude, longitude, time, inacc_radius
    traj = pd.DataFrame(columns = ["lon", "lat", "time", "inacc_radius"])
    traj["lon"] = x_points
    traj["lat"] = y_points
    traj["time"] = t_points
    traj["inacc_radius"] = rad_inacc
    
    return traj

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (in metres) between two points on the earth (specified in decimal degrees)
    
    :param lon1: longitude of point 1
    :param lat1: longitude of point 1
    :param lon2: longitude of point 2
    :param lat2: longitude of point 2
    :return: the distance between (lon1, lat1) and (lon2, lat2), in metres
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6378.1 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def filter(traj, index, filter_type, window_size=3):
    '''
    :param traj: a dataframe representing a series of lat-lon-time data
    :param index: the index of the lat-lon that we want to filter/smoothen
    :param window_size: the number of lat-lon values proceeding and preceeding the indexed lat-lon to consider
    :param filter_type: "median" or "mean" filter
    :return: an array of size (3,1) with lat-lon-time, time is unchanged, lat-lon is based on the filter
    '''
    
    # we don't modify the start and endpoint of our trajectory
    if index == len(traj)-1:
        return traj.iloc[-1,:]
    if index == 0:
        return traj.iloc[0,:]
    
    x_coors = []
    y_coors = []
    
    # collect the datapoints within the desired window
    for i in range(max(0, index-window_size), min(len(traj), index+window_size+1)):
        if i==index:
            continue
        x_coors.append(traj.lon[i])
        y_coors.append(traj.lat[i])
    
    if filter_type == "mean":
        return np.array([mean(x_coors), mean(y_coors), traj.time[index]])
    elif filter_type == "median":
        return np.array([median(x_coors), median(y_coors), traj.time[index]])
    
def getFilteredTraj(traj, filter_type, window_size=3):
    '''
    :param traj: a dataframe representing a series of lat-lon-time data
    :param window_size: the number of lat-lon values proceeding and preceeding the indexed lat-lon to consider
    :param filter_type: "median" or "mean" filter
    :return: a dataframe with filtered lat-lon, the rest of the variables in `traj` dataframe is unchanged,
        filtering of lat-lon is based on the `filter_type`
    '''
    filteredTraj = traj
    filteredTraj = filteredTraj.drop(columns = ["lat", "lon", "time"])
    
    filteredTraj["lon"] = np.nan
    filteredTraj["lat"] = np.nan
    filteredTraj["time"] = np.nan
    
    for i in range(traj.shape[0]):
        filteredPoint = filter(traj, i, filter_type, window_size=3)
        filteredTraj.loc[i, ["lon", "lat", "time"]] = filteredPoint
        
    return filteredTraj

def plot_traj(traj):
    '''
    :param traj: trajectory
    :return:
    plots the trajectory
    '''
    plt.figure(figsize=(8,8))
    plt.title("Trajectory")
    plt.plot(traj.lon, traj.lat, '-bo')
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    
def plot_filtered_traj(traj, mean_traj, median_traj, window_size):
    '''
    :param traj: a dataframe of lat-lon-time and possibly other variables
    :param mean_traj: a dataframe of lat-lon-time and possibly other variables
    :param median_traj: a dataframe of lat-lon-time and possibly other variables
    :param window_size: window size used in the filters
    :return:
        plots the mean filtered trajectory and median filtered trajectory side-by-side
    '''
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    
    plt.suptitle("Filters of window size " + str(window_size))
    
    axs[0].plot(traj.lon, traj.lat, '-bo')
    axs[0].plot(mean_traj.lon, mean_traj.lat, '--ro')
    axs[0].set_title("Randomly generated trajectory\n(mean filtered)")
    axs[0].legend(["Original trajectory", "Mean filtered trajectory"], loc='best')

    axs[1].plot(traj.lon, traj.lat, '-bo')
    axs[1].plot(median_traj.lon, median_traj.lat, '--ro')
    axs[1].set_title("Randomly generated trajectory\n(median filtered)")
    axs[1].legend(["Original trajectory", "Median filtered trajectory"], loc='best')

    for ax in axs.flat:
        ax.set(xlabel='longitude', ylabel='latitude')

class stayPoint:
    def __init__(self, arrivalTime, departTime, startIndex, endIndex, location):
        '''
        :param arrivalTime: The time when the moving object arrived at this stay point.
        :param departTime: The time when the moving object departed this stay point.
        :param startIndex: The index in the object's trajectory dataset corresponding to `arrivalTime`.
        :param endIndex: The index in the object's trajectory dataset corresponding to `departTime`.
        :param location: The [`lon`, `lat`] values corresponding to the location of this stay point.
        '''
        self.arrivalTime = arrivalTime
        self.departTime = departTime
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.location = location
        
    def toString(self):
        '''
        prints all the information about this stay point.
        '''
        print(f"(arrivalTime: {self.arrivalTime}, departTime: {self.departTime}, startIndex: {self.startIndex}, "+
            f"endIndex: {self.endIndex}, location: {self.location})")

def SPDA(traj, distThres, timeThres, minPoints):
    '''
    :param traj: a trajectory
    :param distThres: a threshold of the distance (in metres)
    :param timeThres: a threshold of the time (in seconds)
    :param minPoints: the minimum no. of points required in a stay-point region
    :output: a set of stay-points
    '''
    def distance(pointA, pointB):
        '''
        :param pointA: a point with lat and lon variables
        :param pointB: a point with lat and lon variables
        :return: the distance between pointA and pointB calculated by Haversine formula
        '''
        return haversine(pointA.lon, pointA.lat, pointB.lon, pointB.lat)
    
    def getCentroid(points, centroid_type):
        '''
        :param points: a list of points with lat and lon variables
        :param centroid_type: "median" or "mean"
        :return: the centre of the list of points, calculated by centroid_type function
        '''
#         print("centroid points:\n", points)
        if centroid_type == "median":
            return [median(points.loc[:,"lon"]), median(points.loc[:,"lat"])]
        elif centroid_type == "mean":
            return [mean(points.loc[:,"lon"]), mean(points.loc[:,"lat"])]
        
    i = 0
    pointNum = len(traj)
    stayPoints = []
    
    while i < pointNum:
        j = i+1
        token = 0
        while j < pointNum:
            print("Analysing point: " + str(j) + " "*10, "\r", end="")
            dist = distance(traj.iloc[j,:], traj.iloc[i,:])
            if dist > distThres:
                timeDiff = (traj.time[j] - traj.time[i]).total_seconds()
                if (timeDiff > timeThres) and (j-i >= minPoints):
                    centroid = getCentroid(traj.loc[i:(j-1),:], "median")
                    stayPoints.append(
                        stayPoint(
                            arrivalTime = traj.time[i], 
                            departTime = traj.time[j], 
                            startIndex = i,
                            endIndex = j,
                            location = centroid
                        )
                    )
                    
                    i = j
                    token = 1
                break
            j += 1
            
        if token != 1:
            i += 1
            
    return stayPoints