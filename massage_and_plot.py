import pandas as pd
import matplotlib.pyplot as plt
from get_data import loader

forecast = pd.read_csv("forecasted.csv", index_col=0)
forecast.clip(lower=0, inplace=True)
df = loader()
# chop off a bunch of zeros
df = df[42:]

forecast.plot()
plt.show()
