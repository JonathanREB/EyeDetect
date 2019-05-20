import numpy
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
from sklearn import linear_model, datasets


import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets


def ransac2(data,n,k,t,d,pointX, pointY):
    maybe_idxs, test_idxs = random_partition(n,data.shape[0])
    train_points = data[maybe_idxs,:]
    test_points = data[test_idxs]


    df = pd.DataFrame(data)
    df.columns = ['gaze_X','gaze_Y', 'clic_X', 'clic_Y','side']  

    train_points_df = pd.DataFrame(train_points)
    train_points_df.columns = ['gaze_X','gaze_Y', 'clic_X', 'clic_Y','side']  

    test_points_df = pd.DataFrame(test_points)
    test_points_df.columns = ['gaze_X','gaze_Y', 'clic_X', 'clic_Y','side']    

    gaze_X = np.asarray(df[['gaze_X']])
    gaze_Y = np.asarray(df[['gaze_Y']])
    clic_X = np.asarray(df[['clic_X']])
    clic_Y = np.asarray(df[['clic_Y']])


    gaze_train_X = np.asarray(train_points_df[['gaze_X']])
    gaze_train_Y = np.asarray(train_points_df[['gaze_Y']])
    clic_train_X = np.asarray(train_points_df[['clic_X']])
    clic_train_Y = np.asarray(train_points_df[['clic_Y']])

    gaze_test_X = np.asarray(train_points_df[['gaze_X']])
    gaze_test_Y = np.asarray(train_points_df[['gaze_Y']])
    clic_test_X = np.asarray(train_points_df[['clic_X']])
    clic_test_Y = np.asarray(train_points_df[['clic_Y']])    

    # Activities for X
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(gaze_X, clic_X)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(gaze_X, clic_X)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
#    line_X = np.arange(gaze_test_X.min(), gaze_test_X.max())[:, np.newaxis]
#    line_y = lr.predict(line_X)
#    line_y_ransac = ransac.predict(line_X)

    # Activities for Y
    # Fit line using all data
    lr_Y = linear_model.LinearRegression()
    lr_Y.fit(gaze_Y, clic_Y)

    # Robustly fit linear model with RANSAC algorithm
    ransac_Y = linear_model.RANSACRegressor()
    ransac_Y.fit(gaze_Y, clic_Y)
    inlier_mask_Y = ransac.inlier_mask_
    outlier_mask_Y = np.logical_not(inlier_mask_Y)


    # Predict the coord
    pointY_X = lr.predict(pointX)
    pointY_Y = lr_Y.predict(pointY)
    print("Coordenada de RANSAC:", pointY_X, pointY_Y)
    return pointY_X, pointY_Y