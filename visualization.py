from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import pandas as pd
import datetime as datetime
import scipy
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import hilbert

peak_dist = 60

def create_plot(db, state_name, state_id, county_id=None):

    plt.rcParams["figure.figsize"] = [12.00, 8]
    plt.rcParams["figure.autolayout"] = True

    if county_id is None:
        df = pd.read_sql_query('''
            SELECT tdate, sum(count) AS totals 
            FROM tweet_count 
            LEFT JOIN county ON county_id = county.id
            WHERE state_id=?
            AND tdate > "2020-05-15"
            GROUP BY tdate
        ''', db, params = (state_id,))
    else:
        df = pd.read_sql_query('''
            SELECT tdate, sum(count) AS totals 
            FROM tweet_count 
            LEFT JOIN county ON county_id = county.id
            WHERE state_id=? AND county_ascii=?
            AND tdate > "2020-05-15"
            GROUP BY tdate
        ''', db, params = (state_id, county_id))

    x = pd.to_datetime(df['tdate'])
    y = df['totals'].to_numpy()

    y_hat = savgol_filter(y, 61, 2) # window size 51, polynomial order 3
    
    mean = np.average(y_hat)
    peaks, properties = find_peaks(y_hat, mean, distance=peak_dist)
    # valid = [i for i in peaks if y_hat[i]> mean]
    peak_x = x[peaks]
    peak_y = y_hat[peaks]

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis="x", which="both", rotation=90)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b")) 
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b")) 
    ax.set_ylabel('Tweet Counts (per day)')

    # plt.scatter(peak_x, peak_y, zorder=100, color='red',)
    for px in peak_x:
        plt.axvline(x=px, color='black', linestyle='--')

    # plt.plot(x, y,  color='lightgray')
    line1, = ax.plot(x, y_hat, color='red', label='Tweets')
    plt.grid()
    plt.suptitle(f"{county_id} County, {state_name}")
    plt.title('Query: "fever OR cough"', fontsize=11)


    ax2 = ax.twinx()
    df = pd.read_csv('data/covid_by_county.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['state']==state_name]
    df = df[df['county']==county_id]
    df = df[df['date']>'2020-05-15']
    series = df.groupby(['date'])['new'].sum()
    x = series.index
    y = series.to_numpy().clip(0.01, None)
    y_hat = savgol_filter(y, 61, 4)

    line2, = ax2.plot(x, y_hat, color='blue', label='New Cases')
    ax2.set_ylabel("New Covid Cases (per day)")
    # ax.legend()

    mean = np.average(y_hat)
    peaks, properties = find_peaks(y_hat, 0, distance=peak_dist)
    # valid = [i for i in peaks if y_hat[i]> mean]
    peak_x = x[peaks]
    peak_y = y_hat[peaks]
    # plt.scatter(peak_x, peak_y, zorder=100, color='blue')

    plt.legend([line1, line2], ['Tweets', 'New Cases'])
    # plt.show()
    plt.savefig(f'output/{state_id}-{county_id}.svg', bbox_inches='tight')
    print(county_id)

if __name__ == "__main__":
    db = sqlite3.connect('pandemic.db')
    # for (state_name, state_id) in [('Georgia', 'GA'), ('Florida', 'FL'), ('New York', 'NY'), ('California', 'CA')][3:4]:
    #     create_plot(db, state_name, state_id)
    county_list = [
        ('Georgia', 'GA', 'DeKalb'),
        ('Georgia', 'GA', 'Fulton'),
        ('Georgia', 'GA', 'Cobb'),
        ('Georgia', 'GA', 'Gwinnett'),
    ]
    for (state_name, state_id, county_id) in county_list:
        create_plot(db, state_name, state_id, county_id)
