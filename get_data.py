import sqlite3
import pandas as pd

SOURCE = {
    "IN": ("BEDS_ICU_OCCUPIED_COVID_19", "data/INDIANA_covid_report_bedvent_date.csv"),
    "GA": ("???", "???"),
}


def loader(state="IN"):
    # load twitter data
    db = sqlite3.connect("pandemic.db")
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT tdate, sum(count) as tcount FROM tweet_count
        LEFT JOIN county ON county.id = county_id
        WHERE county.state_id = ?
        GROUP BY tdate
        ORDER BY tdate
    """,
        (state,),
    )
    tweets = pd.DataFrame(
        cursor.fetchall(),
        columns=["DATE", "TWEETS"],
    )
    tweets["DATE"] = pd.to_datetime(tweets.DATE)

    col_name, file_name = SOURCE[state]
    # load dataset
    df = pd.read_csv(
        file_name,
        header=0,
    )
    df["DATE"] = pd.to_datetime(df.DATE)
    df = df[["DATE", col_name]]
    df = pd.merge(tweets, df, how="inner", on="DATE")
    df.set_index("DATE", inplace=True)

    return df


if __name__ == "__main__":
    df = loader()
    print(df.info())
