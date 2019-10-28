import sys
import pickle
import numpy as np
from collections import OrderedDict
from typing import NamedTuple
from pprint import pprint
import matplotlib.pyplot as plt

class Graph():
	'''Class that defines graph/distances'''

	def __init__(self, vertices):
		'''Initialize graph object'''

		# Set of all vertices
		self.V = vertices
	
		# Define dictionary of weights
		self.graph = {}
		for v1 in self.V:
			self.graph[v1] = {}

	def make_connection(self, vertex1, vertex2, weight):
		'''Initialize connection between vertices'''

		# If vertecies in graph
		if (vertex1 in self.V and vertex2 in self.V):
			# Set weights
			self.graph[vertex1][vertex2] = weight
			self.graph[vertex2][vertex1] = weight

	def insert_node(self, i, k, j, l1, l2, np):
		'''Function that inserts new labels (floyd-warshall)'''
		key1 = self.hash_pair(i, k)
		key2 = self.hash_pair(k, j)
		p = np[key1]
		q = np[key2]

		list1_ik = l1[key1]
		list2_ik = l2[key1]
		list1_kj = l1[key2]
		list2_kj = l2[key2]
		
		new1 = []
		new2 = []

		while p > 0 and q > 0:
			new1.insert(0, list1_ik[p-1] + list1_kj[q-1])
			new2.insert(0, max(list2_ik[p-1], list2_kj[q-1]))

			if list2_ik[p-1] > list2_kj[q-1]:
				p -=1
			elif list2_ik[p-1] < list2_kj[q-1]:
				q -= 1
			else:
				p -=1
				q -=1

		return new1, new2

	def max_labels(self, l1, l2, l1_prime, l2_prime):
		'''Function that consolodates labels (floyd-warshall)'''
		p = 0
		q = 0

		best_l = sys.maxsize

		new_l1 = []
		new_l2 = []

		num_labels = 0

		while p < len(l1) and q < len(l1_prime):
			if l2[p] < l2_prime[q]:
				if l1[p] < best_l:
					new_l1.append(l1[p])
					new_l2.append(l2[p])
					best_l = l1[p]
					num_labels += 1
				p += 1
			elif l2[p] > l2_prime[q]:
				if l1_prime[q] < best_l:
					new_l1.append(l1_prime[q])
					new_l2.append(l2_prime[q])
					best_l = l1_prime[q]
					num_labels += 1
				q += 1
			else:
				if l1[p] <= l1_prime[q] and l1[p] < best_l:
					new_l1.append(l1[p])
					new_l2.append(l2[p])
					best_l = l1[p]
					num_labels += 1
				elif l1[p] > l1_prime[q] and l1_prime[q] < best_l:
					new_l1.append(l1_prime[q])
					new_l2.append(l2_prime[q])
					best_l = l1_prime[q]
					num_labels += 1
				p += 1
				q += 1

		while p < len(l1):
			if l1[p] < best_l:
				new_l1.append(l1[p])
				new_l2.append(l2[p])
				best_l = l1[p]
				num_labels += 1
			p += 1
				
			
		while q < len(l1_prime):
			if l1_prime[q] < best_l:
				new_l1.append(l1_prime[q])
				new_l2.append(l2_prime[q])
				best_l = l1_prime[q]
				num_labels += 1
			q += 1

		return new_l1, new_l2, num_labels	

	def hash_pair(self, a, b):
		'''Function that computes hash for pair of vertices'''
		arr = [a, b]
		arr.sort()
		return str(arr)

	def floyd(self):
		'''Floyd-warshall method for all-to-all distances'''
		# Initialize lists
		l1 = {}
		l2 = {}
		np = {}

		# Initialize values for edges
		for v1 in self.V:
			for v2 in self.V:
				if v1 != v2:
					key = self.hash_pair(v1, v2)
					if v2 in self.graph[v1]:
						l1[key] = [1]
						l2[key] = [self.graph[v1][v2]]
						np[key] = 1
					else:
						l1[key] = []
						l2[key] = []
						np[key] = 0
	
		# Iteration for floyd-warshall	
		for k in self.V:
			print(k)
			for i in self.V:
				for j in self.V:
					if i == j or i == k or j == k:
						continue

					key = self.hash_pair(i, j)
					l1_prime, l2_prime = self.insert_node(i, k, j, l1, l2, np)
					l1[key], l2[key], np[key] = self.max_labels(l1[key], l2[key], l1_prime, l2_prime)

		# Lists to store distance/pair
		dist_list = []
		pair_list = {}

		# Store minimum distance/pair data
		for i in self.V:
			for j in self.V:
				if i != j:
					key = self.hash_pair(i, j)
					list1 = l1[key]
					list2 = l2[key]
					if len(list1) != 0:
						dist = [list1[x] * list2[x] for x in range(len(list1))]
						dist_list.append(min(dist))
						pair_list[(i,j)] = (min(dist), list1, list2)

		# Return results
		return dist_list, pair_list

	def debug(self):
		'''Code to debug and compare dijkstra and floyd-warshall methods'''
		# Generate all-to-all distances
		all_to_all = self.floyd()[1]
		one_to_all = {}
		geo = {}
		max_weight = {}
		for source in self.V:
			print(source)
			sw, _, width = self.bottle_neck(source)
			dijk_geo, _ = self.dijkstras_geodesic(source)
			for key in sw:
				if key != source:
					one_to_all[(source, key)] = sw[key]
					geo[(source, key)] = dijk_geo[key]
					max_weight[(source, key)] = width[key]

		# Compare output of both methods, print differences
		for a in self.V:
			for b in self.V:
				key = (a,b)
				if key in one_to_all and (one_to_all[key] != all_to_all[key][0] or False):
					print(str(key))
					print('one-to-all: ' + str(one_to_all[key]))
					print('\tDistance: ' + str(geo[key]))
					print('\tWeight: ' + str(max_weight[key]))
					print('all-to-all: ' + str(all_to_all[key][0]))
					print('\tDistance: ' + str(all_to_all[key][1]))
					print('\tWeight: ' + str(all_to_all[key][2]))
					print()

	def generate_plot(self, filename, s_w, s_g, s_s, method):
		'''Generate plot of weighted vs short-wide vs geodesic distances'''

		# List of calculated distances	
		weighted_distance = []
		geodesic_distance = []
		short_wide_distance = []

		# Variable for debugging
		count = 0

		# Calculate all distances
		for source in self.V:
			print(source)

			# Calculate distances
			dijk_weight, _ = self.dijkstras_weighted(source)
			if method == 'd':
				sw, _, _ = self.bottle_neck(source)
			dijk_geo, _ = self.dijkstras_geodesic(source)

			# Store distances
			for vert in self.V:
				# If node is "disconnected" ignore distance
				if dijk_weight[vert] != sys.maxsize and dijk_weight[vert] != 0:
					# Print if bounds are violated
					if method == 'd' and (round(dijk_weight[vert], 4) > round(sw[vert], 4) or round(sw[vert], 4) > round(dijk_geo[vert], 4)):
						print('Bound Violated')
						print('Weight: ' + str(dijk_weight[vert]))
						print('SW: ' + str(sw[vert]))
						print('Geo: ' + str(dijk_geo[vert]))

					# Store calculated distances
					weighted_distance.append(dijk_weight[vert])
					if method == 'd':
						short_wide_distance.append(sw[vert])
					geodesic_distance.append(dijk_geo[vert])

				# Debugging
				else:
					count += 1

		# Calculate short_wide distance with floyd
		if method == 'f':
			short_wide_distance = self.floyd()[0]

		# Sort calculated distances
		weighted_distance.sort()
		geodesic_distance.sort()
		short_wide_distance.sort()

		#print(short_wide_distance)
		with open('raw_output/' + filename + '/' + method + '_' + 'weighted.pkl', 'wb') as f:
			pickle.dump(weighted_distance, f)

		with open('raw_output/' + filename + '/' + method + '_' + 'geodesic.pkl', 'wb') as f:
			pickle.dump(geodesic_distance, f)

		with open('raw_output/' + filename + '/' + method + '_' + '_' + 'short_wide.pkl', 'wb') as f:
			pickle.dump(short_wide_distance, f)

		# Calculate bins
		w_bin = [x*s_w + .01 for x in range(1 + int(max(weighted_distance)/s_w))]
		g_bin = [x*s_g + .01 for x in range(1 + int(max(geodesic_distance)/s_g))]
		s_bin = [x*s_s + .01 for x in range(1 + int(max(short_wide_distance)/s_s))]

		# Generate histogram
		w_hist = list(np.histogram(weighted_distance, w_bin)[0])
		w_hist /= sum(w_hist)
		g_hist = list(np.histogram(geodesic_distance, g_bin)[0])
		g_hist /= sum(g_hist)
		s_hist = list(np.histogram(short_wide_distance, s_bin)[0])
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

		# Debugging
		#print(len(weighted_distance))
		#print(count)

		# Generate y coordiantes
		y = [(len(weighted_distance) - x)/len(weighted_distance) for x in range(len(weighted_distance))]

		# Plot
		plt.plot(w_bin, w_plot, label='Weighted')
		plt.plot(g_bin, g_plot, label='Geodesic')
		plt.plot(s_bin, s_plot, label='Short Wide')

		plt.ylabel('Survival Function of Distance')
		plt.xlabel('Distance')
		plt.legend(loc='upper right')
		plt.ylim(ymin=0)
		plt.xlim(xmin=0) 

		# Save Plot
		plt.savefig('plots/' + filename + '/' + method + '_' + '_' + str(s_w) + '_' + str(s_g) + '_' + str(s_s) + '.png')

		# Display Plot
		plt.show()

	def min_calc(self, min_found, dist):
		'''Function to determine closest vertex to explore'''

		# Initialize minimum to max int	
		min_now = sys.maxsize

		# Initialize next vertex to explore
		v_next = ''

		# Check the distance to all vertex not explored
		for vertex in self.V:
			if not min_found[vertex] and dist[vertex] < min_now:
				min_now = dist[vertex]
				v_next = vertex

		return v_next

	def dijkstras_weighted(self, src):
		'''Calculated dijkstras using weighted distance'''

		# Distance to vertex
		dist = {}

		# Predecessor to vertex
		pred = {}

		# Track explored vertecies
		min_found = {}

		# Initialize distances/explored
		for v in self.V:
			min_found[v] = False
			dist[v] = sys.maxsize

		# Initialize source vertex
		dist[src] = 0
		pred[src] = None

		# Explore graph
		for count in range(len(self.V)):
			# Determine next vertex to explore
			vertex = self.min_calc(min_found, dist)

			if vertex == '':
				break

			# Mark vertex as found
			min_found[vertex] = True

			# Update distances to adjacent vertices
			for vertex2, weight in self.graph[vertex].items():
				if not min_found[vertex2] and dist[vertex2] > dist[vertex] + weight:
					dist[vertex2] = dist[vertex] + weight
					pred[vertex2] = vertex
	
		return dist, pred
			
	def dijkstras_geodesic(self, src):
		'''Calculated dijkstras using geodesic distance'''

		# Distance to vertex
		dist = {}

		# Predecessor to vertex
		pred = {}

		# Track explored vertecies
		min_found = {}

		# Initialize distances/explored
		for v in self.V:
			min_found[v] = False
			dist[v] = sys.maxsize

		# Initialize source vertex
		dist[src] = 0
		pred[src] = None

		# Explore graph
		for count in range(len(self.V)):
			# Determine next vertex to explore
			vertex = self.min_calc(min_found, dist)

			if vertex == '':
				break

			# Mark vertex as found
			min_found[vertex] = True

			# Update distances to adjacent vertices
			for vertex2, weight in self.graph[vertex].items():
				if not min_found[vertex2] and dist[vertex2] > dist[vertex] + 1:
					dist[vertex2] = dist[vertex] + 1
					pred[vertex2] = vertex

		return dist, pred

	class LabelSet(NamedTuple):
		'''Struct to store label data'''
		index: int
		edges: int
		max_width: float
		distance: float
		node: str
		pred_node: str
		pred_index: int


	def min_label(self, label_list, seen_label):
		'''Function that determines the nearest label to explore'''

		# Initialize minimum distance to large number
		min_dist = float("inf")

		# Initialize next labes
		next_label = None

		# Search label_list for next label to explore
		for label in label_list:
			if label.distance < min_dist and label not in seen_label:
				min_dist = label.distance
				next_label = label

		return next_label

	def bottle_neck(self, src):
		'''Calculate bottleneck distance'''

		# Label list for each vertex
		labels = {}

		# Number of labels at each vertex
		num_labels = {}

		# Set of seen nodes
		seen_node = set()

		# Set of seen labels
		seen_label = set()

		# List of nodes to explore	
		label_list = []

		# Initialize vertex (num_labels/label_list)
		for vertex in self.V:
			labels[vertex] = []
			num_labels[vertex] = 0
		
		# Initialize first label (source)
		src_label = self.LabelSet(0, 0, 0, 0, src, None, -1)	
	
		# Add to vertex label_list/num_labels
		labels[src].append(src_label)
		num_labels[src] = 1

		# Add to list of labels to explore
		label_list.append(src_label)

		# While there is node or label not seen
		while any(node not in seen_node for node in self.V) and any(label not in seen_label for label in label_list):
			# Find label with least cost
			next_label = self.min_label(label_list, seen_label)

			# Mark label/node as seen
			seen_label.add(next_label)
			seen_node.add(next_label.node)
			
			# Create list of potential future labels
			temp_label = []

			# Search adjacent nodes for new labels
			for vertex, weight in self.graph[next_label.node].items():
				# Generate new label
				new_label = self.LabelSet(num_labels[vertex], 
							  next_label.edges + 1, 
							  max(weight, next_label.max_width), 
							  (next_label.edges + 1) * max(weight, next_label.max_width), 
							  vertex, 
							  next_label.node, 
							  next_label.index)
				
				# Check if label is strictly dominated by other labels in vertex label_list
				add = True
				for old_label in labels[vertex]:
					if old_label.edges <= new_label.edges and old_label.max_width <= new_label.max_width:
						add = False

				# Only add label if not strictly dominated
				if add:
					# Add new label
					labels[vertex].append(new_label)
					label_list.append(new_label)

					# Consolidate labels
					labels[vertex][:] = [x for x in labels[vertex] if x.edges <= new_label.edges or x.max_width <= new_label.max_width]
					label_list[:] = [x for x in label_list if x.node != vertex or (x.edges <= new_label.edges or x.max_width <= new_label.max_width)]

					# Update number of labels
					num_labels[vertex] = len(labels[vertex])

		# Determine distance/predecessor for each node	
		dist = {}
		pred = {}
		width = {}
		for node in self.V:
			min_dist = float("inf")
			for label in labels[node]:
				if label.distance < min_dist:
					min_dist = label.distance
					pred[node] = (label.pred_node, label.pred_index)
					dist[node] = label.distance
					width[node] = label.max_width
		
		return dist, pred, width
		

if __name__ == "__main__":
	test = Graph({'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'})
	test.make_connection('A', 'B', 0.5)			
	test.make_connection('A', 'E', 1.0)			
	test.make_connection('B', 'C', 0.5)			
	test.make_connection('B', 'D', 0.167)			
	test.make_connection('C', 'F', 0.167)			
	test.make_connection('D', 'F', 0.2)			
	test.make_connection('E', 'G', 1.0)			
	test.make_connection('F', 'H', 0.5)			
	test.make_connection('G', 'H', 1.0)			
	test.make_connection('H', 'I', 0.5)			
	test.make_connection('I', 'J', 0.5)			
	test.make_connection('J', 'K', 1.0)

	test.generate_plot('test', 0.75, 0.9, 0.9, 'f')

	#test.debug()
	#print(test.floyd())

	#dijk_weight_d, dijk_weight_p = test.dijkstras_weighted('A')
	#print('Shortest Path (Dijkstras Weighted):')
	#pprint(dijk_weight_d)
	#print('Predecessor (Dijkstras):')
	#pprint(dijk_weight_p)

	#sw_dist, sw_pred = test.bottle_neck('A')
	#print('\n\nShortest Path (Short Wide):')
	#pprint(sw_dist)
	#print('Predecessor (Short Wide):')
	#pprint(sw_pred)

	#dijk_geo_w, dijk_geo_p = test.dijkstras_geodesic('A')
	#print('\n\nShortest Path (Dijkstras Geodesic):')
	#pprint(dijk_geo_w)
	#print('Predecessor (Dijkstras):')
	#pprint(dijk_geo_p)
