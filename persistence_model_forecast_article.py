# see https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


def parser(x):
    return datetime.strptime("190" + x, "%Y-%m")


# load dataset
series = read_csv(
    "data/shampoo.csv",
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # make prediction
    predictions.append(history[-1])

    # observation
    history.append(test[i])

# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print("RMSE: %.3f" % rmse)

# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
