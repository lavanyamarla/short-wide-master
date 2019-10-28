import sys
import numpy as np
from scipy import sparse
from scipy import io
from pprint import pprint
from algo import Graph
from connected import connected_component

if sys.argv[1] == 'neuron':
	# Open matlab workspaces
	data = io.loadmat('data/processed/neuron_data.mat')

	# Select variable of importance
	graph = data['Ag_t_ordered']
else:
	# Load sparse matrix
	graph = sparse.load_npz('data/processed/' + sys.argv[1] + '.npz')

# Get largest connected component
graph = connected_component(graph)

# Modify edgeweights
np.reciprocal(graph.data, out=graph.data)

# Convert graph to dictionary
graph = dict(graph.todok())

# Store nodes
nodes = [node for tup in graph.keys() for node in tup]
nodes = set(nodes)

# Initialize graph
graph_algo = Graph(nodes)

# Create edges
for key, value in graph.items():
	graph_algo.make_connection(key[0], key[1], value)

# Alert that we have created graph
print('Created Graph')

# Branch for debugging vs plot generation
if len(sys.argv) == 3 and sys.argv[2] == 'debug':
	graph_algo.debug()
else:
	# Generate plot
	graph_algo.generate_plot(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])
