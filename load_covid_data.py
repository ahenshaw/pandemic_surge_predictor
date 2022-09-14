import pandas as pd

def load_covid(state_name, county_id):
    df = pd.read_csv('data/covid_by_county.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['state']==state_name]
    df = df[df['county']==county_id]
    df = df[df['date']>'2020-05-15']
    series = df.groupby(['date'])['new'].sum()
    x = series.index
    y = series.to_numpy()
    return x, y
