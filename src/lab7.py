import numpy as np
import os
import pandas as pd
import re
from typing import List

import typing
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, load_model

import matplotlib.pyplot as plt
from tqdm import tqdm


def get_deltaS_deltaTheta_from_xy(x: np.ndarray, y: np.ndarray,
                                  returnDeltaTheta=True, returnTrig=False,
                                  backFillTheta=False, backFillDeltaD=False) -> np.ndarray:
    """Converts a sequence of x, y positions to relative distance between sequence steps.
    Can also return the relative angle between sequence steps if desired
    Can also return the sin and cos of the relative angle if desired.

    By default the return sequence is the same length as the input sequence but the first timestep is padded to be zero
    If desired the relative distance and/or angle can be backfilled (index 1 is copied to index 0)

    :param x: ndarray of the x positions. The actual data should be in the last dimension.
    :param y: ndarray of the y positions. The actual data should be in the last dimension.
    :param returnDeltaTheta: if true will return the relative angle
    :param returnTrig: if true will return the sin and cos if the relative angle
    :param backFillTheta: if true will make the first relative angle be equal to the second
    :param backFillDeltaD: if true will make the first relative distance equal to the second
    :return: numpy ndarray with relative angle and distance between time steps.
    This will be same length as the input x,y.
    The actual data is stored in the last dimension in the order:
    [delta_d, delta_theata, sin(delta_theta), cos(delta_theta), theta)]
    The first values either set to zero or back filled if back fill is set.
    :rtype: np.ndarray
    """
    reduceTo2D = False
    if x.ndim != y.ndim:
        raise ValueError("x and y should have same dimensions not {0} and {1} respectively".format(x.ndim, y.ndim))
    if x.ndim < 2:
        x = x[None, ...]
        y = y[None, ...]
        reduceTo2D = True

    delta_x = np.diff(x)
    delta_y = np.diff(y)

    # get absolute angle between each diff
    theta = np.concatenate([np.zeros(shape=(delta_y.shape[0], 1)), np.arctan2(delta_y, delta_x)], axis=-1)

    # back fill if necessary
    if backFillTheta:
        theta[..., 0] = theta[..., 1]

    # get the relative distance and angle
    delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2)
    delta_theta = np.diff(theta)

    # get the trig values of the relative angles
    sinDeltaTheta = np.sin(delta_theta)
    cosDeltaTheta = np.cos(delta_theta)

    deltas = np.concatenate(
        [delta_s[..., None], delta_theta[..., None], sinDeltaTheta[..., None], cosDeltaTheta[..., None]], axis=-1)

    # pad the beginning with zeros to match original size
    zerosShape = list(deltas.shape)
    zerosShape[-2] = 1
    deltas = np.concatenate([np.zeros(zerosShape), deltas], axis=-2)
    # the first cosine term should be 1 not zero
    deltas[..., 0, 3] = 1

    if backFillDeltaD:
        deltas[..., 0, 0] = deltas[..., 1, 0]

    # only return what they asked for
    retColumns = [0]
    if returnDeltaTheta:
        retColumns += [1]
    if returnTrig:
        retColumns += [2, 3]
    ret = deltas[..., retColumns]

    if reduceTo2D:
        ret = ret[0]
    return ret


def getXYfromDeltas(delta_d: np.ndarray,
                    delta_theta: np.ndarray,
                    initial_conditions: np.ndarray = None) -> np.ndarray:
    """Converts a sequence of relative distances and angle changes to 2D x,y positions
    Assumes the starting location is 0,0 unless initial_conditions are given

    :param delta_d: relative distances between each step in the sequence
    :param delta_theta: relative angles betweeen each step in the sequence
    :param initial_conditions: the starting x,y point(s)
    :return: a numpy ndarray with the x,y positions in the last dimension
    :rtype: np.ndarray
    """
    reduceTo2D = False
    if delta_d.ndim != delta_theta.ndim:
        raise ValueError(
            "delta_d and delta_theta should have same dimensions not {0} and {1} respectively".format(delta_d.ndim,
                                                                                                      delta_theta.ndim))
    if delta_theta.ndim < 2:
        delta_theta = delta_theta[None, ...]
        delta_d = delta_d[None, ...]
        reduceTo2D = True
    if initial_conditions is None:
        initial_conditions = np.zeros(shape=(delta_d.shape[0], 3))

    theta = np.concatenate([np.zeros(shape=(delta_theta.shape[0], 1)), np.cumsum(delta_theta, axis=-1)], axis=-1)
    theta += initial_conditions[..., 2:3]

    delta_x = delta_d * np.cos(theta[..., :-1] + delta_theta)
    delta_y = delta_d * np.sin(theta[..., :-1] + delta_theta)

    x = np.cumsum(delta_x, axis=-1) + initial_conditions[..., 0:1]
    y = np.cumsum(delta_y, axis=-1) + initial_conditions[..., 1:2]

    position = np.concatenate([x[..., None], y[..., None]], axis=-1)

    if reduceTo2D:
        position = position[0]
    return position


def get_samples_from_lists(imuList: List[pd.DataFrame],
                           viList: List[pd.DataFrame],
                           num_samples: int,
                           seq_len: int,
                           input_columns=None,
                           output_columns=None) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Take the raw data and break up into smaller sequences
    the dataframes in the lists will be 2D (timestep, data)
    the output numpy arrays will be 3D (sample, timestep, data)

    :param imuList: the list of pandas dataframes for the imu data
    :param viList: the list of pandas dataframes for the vi data
    :param num_samples: the total number of samples to return
    :param seq_len: the number of time steps in each sequence
    :param input_columns: the columns to use as input
    :param output_columns: the columns to use as output
    :return:
    """

    # process the lists of pandas data frames (or whatever type you want) into the sequences to train
    x_data: np.ndarray
    y_data: np.ndarray
    return x_data, y_data


def build_model(seq_len,
                input_data_size,
                output_data_size,
                stateful=False) -> Model:
    model: Model
    return model


def main():
    fileRoot = os.path.join("/opt", "data", "Oxford Inertial Tracking Dataset")

    # print our column names that we got from the top level description
    viconColumnNames = "Time Header translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w".split(
        ' ')
    print(viconColumnNames)
    imuColumnNames = "Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)".split(
        ' ')
    print(imuColumnNames)

    # make our variables for the experiment
    input_columns = []
    output_columns = []
    model_name = "model.h5"
    num_samples: int
    seq_len: int

    # read in our raw data
    ignore_first = 2000
    viListTrain = []
    imuListTrain = []
    viListTest = []
    imuListTest = []
    # walk through the oxford directory
    for root, dirs, files in os.walk(fileRoot, topdown=False):
        # we only want the handheld data with synchronized data
        if 'handheld' in root and 'syn' in root:
            # we are going to loop for the number of actual data files (they come in pairs)
            for i in range(len(files) // 2):
                # the name of the mobile phone data (imu) and the vicon truth data (vi)
                imuName = os.path.join(root, f"imu{i + 1}.csv")
                viName = os.path.join(root, f"vi{i + 1}.csv")
                if os.path.exists(viName) and os.path.exists(imuName):
                    # read the csv files into pandas data frames
                    viTemp = pd.read_csv(viName, names=viconColumnNames)[ignore_first:]
                    imuTemp = pd.read_csv(imuName, names=imuColumnNames)[ignore_first:]

                    # convert the x,y position to distance and angles
                    deltas = get_deltaS_deltaTheta_from_xy(viTemp['translation.x'].values,
                                                           viTemp['translation.y'].values)
                    # add distance and angles to data frame
                    viTemp['delta_s'] = deltas[..., 0]
                    viTemp['delta_theta'] = deltas[..., 1]

                    # put the data frame in the correct list
                    dataNum = int(re.search(r"data(?P<num>\d+)", root).group("num"))
                    if dataNum < 5:
                        viListTrain.append(viTemp)
                        imuListTrain.append(imuTemp)
                    else:
                        viListTest.append(viTemp)
                        imuListTest.append(imuTemp)

    print(f"Read {len(viListTrain)} data frames")

    # turn the raw data into sequences
    _, _ = get_samples_from_lists(imuList=imuListTrain,
                                  viList=viListTrain,
                                  num_samples=num_samples,
                                  seq_len=seq_len,
                                  input_columns=input_columns,
                                  output_columns=output_columns)

    input_data_size: int
    output_data_size: int

    # do any other pre-processing
    if not os.path.exists(model_name):
        # build and save model
        model: Model = build_model(seq_len,
                                   input_data_size,
                                   output_data_size,
                                   stateful=False)
        model.save(model_name)
    else:
        model = load_model(model_name)

    # build a model that can take very long sequences (or process one step at a time) for our visualization
    # to make a model remember from one predict call to the next use the `stateful` property of the RNN layers
    model_one_step: Model = build_model(seq_len,
                                        input_data_size,
                                        output_data_size,
                                        stateful=True)

    # make our test data (use training/validation at first) and then create predictions from it
    # remember at this point we are working with longer sequences
    # models need to be able to predict for as many timesteps as required and thus predicting on one timestep at a time works well
    x_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray

    # Visualization

    # print the whole values
    plt.figure()
    # plot our true delta_s values

    # plot our predicted delta_s values

    # label axes
    plt.title('delta_s Whole Values')
    plt.xlabel('timestep')
    plt.ylabel('distance (m)')
    plt.legend()

    # print the top down view
    plt.figure()
    # plot the true position

    # plot our predicted position

    # label axes
    plt.title('Top Down View')
    plt.xlabel('vicon x axis (m)')
    plt.ylabel('vicon y axis (m)')
    plt.legend()


if __name__ == "__main__":
    main()
