from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import pandas as pd
import datetime as datetime
from smoothing import savitzky_golay
from scipy.signal import find_peaks_cwt
import numpy as np

db = sqlite3.connect('pandemic.db')
df = pd.read_sql_query('SELECT tdate, sum(count) AS totals FROM tweet_count GROUP BY tdate', db)

x = pd.to_datetime(df['tdate'])
y = df['totals'].to_numpy()
y_hat = savitzky_golay(y, 61, 3) # window size 51, polynomial order 3
mean = np.average(y_hat)
peaks = find_peaks_cwt(y_hat, 15)
valid = [i for i in peaks if y_hat[i]> mean]
peak_x = x[valid]
peak_y = y_hat[valid]

fig = plt.figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis="x", which="both", rotation=90)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b")) 
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b")) 
ax.set_ylabel('Tweet Counts')

plt.scatter(peak_x, peak_y, zorder=100)
plt.plot(x, y,  color='lightgray')
plt.plot(x,y_hat, color='red')
plt.grid()
plt.suptitle('Georgia')
plt.title('Query: "fever OR cough"', fontsize=11)

ax2 = ax.twinx()
df = pd.read_csv('data/covid_by_county.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['state']=='Georgia']
series = df.groupby(['date'])['new'].sum()
x = series.index
y = series.to_numpy()
y_hat = savitzky_golay(y, 61, 4)
ax2.plot(x, y_hat)
plt.show()