# see https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


def parser(x):
    return datetime.strptime("190" + x, "%Y-%m")


# load dataset
df = read_csv(
    "data/INDIANA_covid_report_bedvent_date.csv",
    header=0,
    # parse_dates=[0],
    index_col=0,
    # squeeze=True,
)
series = df["BEDS_ICU_OCCUPIED_COVID_19"]
# split data into train and test
X = series.values
cutoff = len(X) // 3
train, test = X[0:cutoff], X[cutoff:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # make prediction
    predictions.append(history[-7])

    # observation
    history.append(test[i])

# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print("RMSE: %.3f" % rmse)

# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
