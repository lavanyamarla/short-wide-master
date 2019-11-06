import random
import sys
import pickle
import networkx as nx
from scipy import sparse
from algo import Graph
from connected import connected_component

def create_graph():
	# Load sparse matrix
	graph = sparse.load_npz('data/processed/neuron.npz')

	# Get largest connected component
	graph = connected_component(graph)

	deg = []
	for i in range(248):
        	deg.append(graph[i,:].count_nonzero())

	deg[0] += 1
	deg.sort()

	G = nx.configuration_model(deg,create_using=nx.Graph())

	weights = {}

	for edge in G.edges():
		weights[edge] = random.random()

	nx.set_edge_attributes(G, weights, 'weight')

	G = nx.adjacency_matrix(G)

	return G

if __name__ == "__main__":
	dijk = []
	geo = []
	weight = []

	for i in range (100):
		print(i)
		graph = create_graph()
	
		# Get largest connected component
		graph = connected_component(graph)

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

		w, g, d = graph_algo.generate_distances()
		weight.append(w)
		geo.append(g)
		dijk.append(d)
	
		if (i + 1) % 5 == 0:
			with open('raw_output/deg/weighted.pkl', 'wb') as f:
				pickle.dump(weight, f)

			with open('raw_output/deg/geodesic.pkl', 'wb') as f:
				pickle.dump(geo, f)

			with open('raw_output/deg/sw_d.pkl', 'wb') as f:
				pickle.dump(dijk, f)

	with open('raw_output/deg/weighted.pkl', 'wb') as f:
		pickle.dump(weight, f)

	with open('raw_output/deg/geodesic.pkl', 'wb') as f:
		pickle.dump(geo, f)

	with open('raw_output/deg/sw_d.pkl', 'wb') as f:
		pickle.dump(dijk, f)

	weighted_distance = weight
	geodesic_distance = geo
	short_wide_distance = dijk

	weighted_distance.sort()
	geodesic_distance.sort()
	short_wide_distance.sort()

	weighted_distance.append(weighted_distance.pop(int(len(weighted_distance)/2)))
	geodesic_distance.append(geodesic_distance.pop(int(len(weighted_distance)/2)))
	short_wide_distance.append(short_wide_distance.pop(int(len(weighted_distance)/2)))

	s_w = float(sys.argv[1])
	s_g = float(sys.argv[2])
	s_s = float(sys.argv[3])

	w_b = []
	w_p = []
	g_b = []
	g_p = []
	s_b = []
	s_p = []

	for i in range(len(weighted_distance)):
        	# Calculate bins
        	w_bin = [x*s_w + .01 for x in range(1 + int(max(weighted_distance[i])/s_w))]
        	g_bin = [x*s_g + .01 for x in range(1 + int(max(geodesic_distance[i])/s_g))]
        	s_bin = [x*s_s + .01 for x in range(1 + int(max(short_wide_distance[i])/s_s))]
        	w_b.append(w_bin)
        	g_b.append(g_bin)
        	s_b.append(s_bin)

	        # Generate histogram
        	w_hist = list(np.histogram(weighted_distance[i], w_bin)[0])
        	w_hist /= sum(w_hist)
        	g_hist = list(np.histogram(geodesic_distance[i], g_bin)[0])
        	g_hist /= sum(g_hist)
        	s_hist = list(np.histogram(short_wide_distance[i], s_bin)[0])
        	s_hist /= sum(s_hist)

	        # Generate survival
        	w_plot = []
        	for i in range(len(w_hist)):
                	w_plot.append(sum(w_hist[i:]))
        	w_plot.append(0)

        	g_plot = []
        	for i in range(len(g_hist)):
        	        g_plot.append(sum(g_hist[i:]))
        	g_plot.append(0)

        	s_plot = []
        	for i in range(len(s_hist)):
                	s_plot.append(sum(s_hist[i:]))
        	s_plot.append(0)
        	w_p.append(w_plot)
        	g_p.append(g_plot)
        	s_p.append(s_plot)

	for i in range(len(weighted_distance)):
        	# Plot
        	if i == len(weighted_distance) - 1:
                	plt.plot(w_b[i], w_p[i], color='b', label='Weighted')
                	plt.plot(g_b[i], g_p[i], color='g', label='Geodesic')
                	plt.plot(s_b[i], s_p[i], color='r', label='Short Wide')
        	else:
                	plt.plot(w_b[i], w_p[i], color='0.75', alpha=0.5)
                	plt.plot(g_b[i], g_p[i], color='0.75', alpha=0.5)
                	plt.plot(s_b[i], s_p[i], color='0.75', alpha=0.5)

	plt.ylabel('Survival Function of Distance')
	plt.xlabel('Distance')
	plt.legend(loc='upper right')
	plt.ylim(ymin=0)
	plt.xlim(xmin=0, xmax=8)

	# Save Plot
	plt.savefig('plots/deg/plot.png')

	# Display Plot
	plt.show()
