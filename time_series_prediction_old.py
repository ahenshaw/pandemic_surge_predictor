from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from load_covid_data import load_covid

date, new_cases = load_covid('Georgia', 'Fulton')
df = pd.DataFrame(dict(date=date, new_cases=new_cases))
df = df.set_index('date')
scaler = MinMaxScaler()

cutoff = int(0.8 * len(df))
train = df.iloc[:cutoff]
test  = df.iloc[cutoff:]

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test  = scaler.transform(test)

# our input window size (in days)
window = 14
batch_size = 1
num_features = 1 # 2 when adding the tweet data
num_neurons = 10

generator = TimeseriesGenerator(scaled_train, scaled_train, length=window, batch_size=batch_size)

print('Building model...')
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(window, num_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
model.fit(generator, epochs=50)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)