from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import pandas as pd
import datetime as datetime
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
import numpy as np

def create_plot(db, state_name, state_id):
    df = pd.read_sql_query('''
        SELECT tdate, sum(count) AS totals 
        FROM tweet_count 
        LEFT JOIN county ON county_id = county.id
        WHERE state_id=?
        GROUP BY tdate
    ''', db, params = (state_id,))

    x = pd.to_datetime(df['tdate'])
    y = df['totals'].to_numpy()
    y_hat = savgol_filter(y, 61, 3) # window size 51, polynomial order 3
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
    plt.suptitle(state_name)
    plt.title('Query: "fever OR cough"', fontsize=11)

    ax2 = ax.twinx()
    df = pd.read_csv('data/covid_by_county.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['state']==state_name]
    series = df.groupby(['date'])['new'].sum()
    x = series.index
    y = series.to_numpy()
    y_hat = savitzky_golay(y, 61, 4)
    ax2.plot(x, y_hat)
    plt.show()

if __name__ == "__main__":
    db = sqlite3.connect('pandemic.db')
    for (state_name, state_id) in [('Georgia', 'GA'), ('Florida', 'FL'), ('New York', 'NY'), ('California', 'CA')]:
        create_plot(db, state_name, state_id)

