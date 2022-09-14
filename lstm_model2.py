import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
from matplotlib import pyplot as plt
import sqlite3

from get_data import loader

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.iloc[:, -1].shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars - 1, n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars - 1, n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg, n_vars


# create a differenced dataframe
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff)


# transform series into train and test sets for supervised learning
def prepare_data(df, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = df.values

    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values

    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)

    # transform into supervised learning problem X, y
    supervised, n_vars = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values

    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test, n_vars


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons, n_vars):
    # reshape training into [samples, timesteps, features]
    lagged_vars = n_lag * n_vars
    X, y = train[:, 0:lagged_vars], train[:, lagged_vars:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(
        LSTM(
            n_neurons,
            batch_input_shape=(n_batch, X.shape[1], X.shape[2]),
            stateful=True,
        )
    )
    model.add(Dense(y.shape[1]))
    model.compile(loss="mean_squared_error", optimizer="adam")
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to np.array
    return [x for x in forecast[0, :]]


# evaluate the LSTM model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq, n_vars):
    forecasts = list()
    lagged_vars = n_vars * n_lag
    for i in range(len(test)):
        X, y = test[i, 0:lagged_vars], test[i, lagged_vars:]
        print(X.shape, y.shape)
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    sc = MinMaxScaler()
    sc.min_, sc.scale_ = scaler.min_[1], scaler.scale_[1]
    inverted = list()
    for i in range(len(forecasts)):
        # create np.array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = sc.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        print(i, "actual", actual)
        print(i, "predic", predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        print("t+%d RMSE: %f" % ((i + 1), rmse))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test, n_seq):

    ref = series.values[:, -1]
    print(ref)
    plt.plot(ref)

    fx = []
    fy = []
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(ref) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x - n_seq + 1 for x in range(off_s, off_e)]
        yaxis = [ref[off_s]] + forecasts[i]
        fx.append(xaxis[-1])
        fy.append(yaxis[-1][-1])
        # pyplot.plot(xaxis, yaxis, color="red")
    plt.plot(fx, fy, color="red")
    # show the plot
    plt.show()


def save_forecast(df, forecast, test, n_seq):
    forecast = pd.Series([x[-1][-1] for x in forecast], name="FORECAST")
    forecast.to_csv("forecasted.csv")


if __name__ == "__main__":
    import sys

    df = loader()

    series = df["BEDS_ICU_OCCUPIED_COVID_19"]
    # chop off a bunch of zeros at the beginning
    # series = series[42:]
    df = df[42:]

    # configure
    n_lag = 1
    n_seq = 7
    n_test = len(series) // 2
    n_epochs = 10
    n_batch = 1
    n_neurons = 1
    # prepare data
    scaler, train, test, n_vars = prepare_data(df, n_test, n_lag, n_seq)
    print("train", train.shape, n_vars)

    # fit model
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons, n_vars)

    # make forecasts
    forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq, n_vars)

    # inverse transform forecasts and test
    forecasts = inverse_transform(df, forecasts, scaler, n_test + n_seq - 1)
    save_forecast(df, forecasts, test, n_seq)
    sys.exit()
    actual = [row[n_lag * n_vars :] for row in test]
    actual = inverse_transform(df, actual, scaler, n_test + n_seq - 1)

    # evaluate forecasts
    # evaluate_forecasts(actual, forecasts, n_lag, n_seq)

    # plot forecasts
    plot_forecasts(df, forecasts, n_test + n_seq - 1, n_seq)
