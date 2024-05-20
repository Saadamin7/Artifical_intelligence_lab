import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('E:/flights.csv/flights.csv', dtype={'ORIGIN_AIRPORT': str, 'DESTINATION_AIRPORT': str})

# Step 2: Process the data to create a graph representation
G = nx.Graph()

for index, row in data.iterrows():
    origin = row['ORIGIN_AIRPORT']
    destination = row['DESTINATION_AIRPORT']
    origin_short = origin[:3]  # Extract first three characters as short name
    destination_short = destination[:3]  # Extract first three characters as short name
    departure_delay = abs(row['DEPARTURE_DELAY'])
    arrival_delay = abs(row['ARRIVAL_DELAY'])
    weight = departure_delay + arrival_delay
    distance = row['DISTANCE']
    airtime = row['AIR_TIME']
    
    if G.has_edge(origin_short, destination_short):
        # Update edge attributes for existing edge
        G[origin_short][destination_short]['weight'] += weight
    else:
        # Create new edge with attributes
        G.add_edge(origin_short, destination_short, weight=weight, distance=distance, airtime=airtime)

# Step 3: Analyze the graph
# 4) List 3 highest degree nodes and 3 lowest degree nodes
degree_dict = dict(G.degree())
highest_degree_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:3]
lowest_degree_nodes = sorted(degree_dict.items(), key=lambda x: x[1])[:3]
print("Highest degree nodes:", highest_degree_nodes)
print("Lowest degree nodes:", lowest_degree_nodes)

# Step 4: Visualize the graph
degrees = [degree_dict[node] for node in G.nodes()]
nodes = list(G.nodes())

plt.figure(figsize=(10, 10))
nx.draw(G, pos=nx.circular_layout(G), node_color=degrees, cmap=plt.cm.Blues, with_labels=True, node_size=1000)
plt.title("Flight Network Visualization with Circular Layout and Node Color by Degree")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), label="Degree")
plt.axis('off')
plt.show()