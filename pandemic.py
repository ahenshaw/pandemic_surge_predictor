import sys
import requests
import json
from datetime import datetime
import sqlite3
from yaspin import yaspin

from secrets import BEARER_TOKEN


def search_twitter(params):    
    
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    url = "https://api.twitter.com/2/tweets/counts/all"

    pages = []
    while True:
        response = requests.request("GET", url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f'\n==== {response.reason} ====')
            sys.exit()
            
        json = response.json()
        pages.append(json)

        # If the response includes a "next_token", then we need to request the next page.
        # Otherwise, we're done for now
        try:
            next_token = json['meta']['next_token']
            params['pagination_token'] = next_token
        except KeyError:
            break
    return pages

def get_tweets_from_near_location(keywords, lat, lon, radius=25, start_time='2020-01-01T00:00:00Z', end_time='2022-03-30T00:00:00Z'):
    params = {
        'query': f'({keywords}) point_radius:[{lon} {lat} {radius}mi]',
        'granularity': 'day', 
        'start_time': start_time,
        'end_time': end_time
    }
    pages = search_twitter(params)
    results = []
    for page in pages:
        try:
            data = page['data']
            results.extend([(datetime.strptime(x['start'], '%Y-%m-%dT%H:%M:%S.000Z'), x['tweet_count']) for x in data])
        except KeyError:
            pass
    return results


if __name__ == '__main__':
    KEYWORDS = 'cough OR fever'

    db = sqlite3.connect('pandemic.db')
    cursor = db.cursor()

    # We want to ignore counties that have already been recorded, so make a set
    cursor.execute('SELECT DISTINCT county_id FROM tweet_count')
    seen = set([x[0] for x in cursor.fetchall()])
    
    count_fetched = 1
    cursor.execute('SELECT id, lat, lng, county_ascii, state_id from county where state_id="GA" order by state_id, county_ascii')
    for county_id, lat, lon, county_name, state in cursor.fetchall():
        if county_id not in seen:
            with yaspin(text=f'{count_fetched}. {county_name}, {state}') as spinner:
                results = get_tweets_from_near_location(KEYWORDS, lat, lon)
                spinner.ok("✅ ")
            to_insert = []
            for date, count in results:
                if count:
                    to_insert.append((county_id, date, count))
            cursor.executemany('INSERT INTO tweet_count (county_id, tdate, count) VALUES (?, ?, ?)', to_insert)
            db.commit()
            count_fetched += 1
        else:
            pass
            # print(f'Skipping {county_name} {state}')
        