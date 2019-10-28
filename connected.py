import scipy.io

def connected_component(graph):
	'''Function that returns largest connected component'''
	dim = graph.shape[0]
	
	seen = set()
	connected = []

	for node in range(dim):
		if node not in seen:
			to_explore = [node]
			sub_graph = set()
			while len(to_explore) != 0:
				current_node = to_explore.pop()
				if current_node in seen:
					continue
				seen.add(current_node)
				sub_graph.add(current_node)
				for i in graph[current_node].tocoo().col:
					if i not in seen:
						to_explore.append(i)
			connected.append(sub_graph)
		else:
			continue

	max_len = 0
	for lis in connected:
		if len(lis) > max_len:
			max_len = len(lis)
			max_com = lis
	max_com = list(max_com)

	return graph[max_com,:][:,max_com]
