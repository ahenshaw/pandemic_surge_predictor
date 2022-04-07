import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
from itertools import permutations
from math import log10
import numpy as np
import great_circle_calculator.great_circle_calculator as gcc
import utm

cmap = get_cmap('OrRd')
CUTOFF = 10000

class Analyst:
    def __init__(self):
        df = pd.read_csv('data/GEORGIA.csv')

        # create a dataframe to just hold the general info for each county
        counties = df.groupby(['county_ascii']).first().reset_index()
        self.counties = counties.set_index('county_ascii')

        # this is the dataframe that we'll use to identify the counties with increasing cases
        self.cases = df.groupby(['county_ascii', 'time_stamp']).first().reset_index()
        self.cases = self.cases.set_index('county_ascii')
        self.cases['delta'] = self.cases['count_cases'].diff()
        self.max_log_diff = log10(self.cases['delta'].max())

        # # make one node for each county (active or not)
        self.graph = nx.Graph()
        for index, data in self.counties.iterrows():
            x, y, _, _ = utm.from_latlon(data['lat'], data['lng'])
            self.graph.add_node(index, pos = (data['lng'], data['lat']), label=index)

        self.compute_distances()

    def compute_distances(self):
        earth_radius = 3959 # in miles
        self.grid = {}

        c = self.counties  # make a shortened alias
        for (name1, d1), (name2, d2) in permutations(c.iterrows(), 2):
            self.grid[(name1, name2)] = gcc.distance_between_points((d1.lng, d1.lat), (d2.lng, d2.lat), unit='miles')
    
    def get_weeks(self):
        return sorted(self.cases['time_stamp'].unique())

    def make_plot(self, target_week):
        # get a fresh copy of our counties graph
        g = nx.create_empty_copy(self.graph)

        # extract the counties that are active for the target week
        active = self.cases[self.cases['delta'] > 0]
        active = active[active['time_stamp'] == target_week]

        # just need county names for the active counties
        # we make it a set so that we can quickly test membership later on
        active_set = set(active.index.tolist())

        for a, b in permutations(active_set, 2):
            if self.grid[(a, b)] < CUTOFF:
                g.add_edge(a, b)

        color_map = []
        for node in g:
            if node not in active_set:
                color_map.append('lightgray')
            else:
                color = cmap(log10(active.loc[[node]]['delta']) / self.max_log_diff)
                color_map.append(rgb2hex(color))

        pos = nx.get_node_attributes(g, 'pos')
        plt.clf()
        nx.draw(g, pos, node_color=color_map, with_labels=True, font_size=10)

        # bit hacky, but seems to be the best way to set the edge color of a node
        ax = plt.gca() # to get the current axis
        # ax.collections[0].set_edgecolor("black") 
        # ax.collections[0].set_linewidth(1) 

        figure = plt.gcf()
        figure.set_size_inches(12, 12)

        plt.title(f'Week {target_week}')
        plt.savefig(f'output/ga{target_week:03}.png', dpi=120)



analyst = Analyst()
weeks = analyst.get_weeks()
for week in weeks:
    print(f'Week {week} of {weeks[-1]}')
    # output goes into an 'output' folder
    analyst.make_plot(target_week=week)

# use this command-line program to make into a movie
# ffmpeg -r 1 -start_number 7 -i ga%03d.png -vcodec mpeg4 -y ga.mp4
