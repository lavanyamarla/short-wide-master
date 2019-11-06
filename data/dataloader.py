import sys
import numpy as np
from scipy import sparse


# Open indian bus file (mbn)
if sys.argv[1] == 'facebook':
	with open('raw/facebook/facebook_combined.txt') as f:
		data = f.read()

	dim = 4039

	data = data.split('\n')[:-1]

	delimeter = ' '

# Open indian bus file (mbn)
if sys.argv[1] == 'india_mbn':
	with open('raw/india/mbn.txt') as f:
		data = f.read()

	dim = 2267

	data = data.split('\n')[1:]

	delimeter = '\t'

# Open indian bus file (cbn)
if sys.argv[1] == 'india_cbn':
	with open('raw/india/cbn.txt') as f:
		data = f.read()

	dim = 1218

	data = data.split('\n')[1:]

	delimeter = '\t'

# Open US airport file
if sys.argv[1] == 'airport':
	with open('raw/US_airports_2010.txt') as f:
		data = f.read()

	dim = 1858

	data = data.split('\n')[4:]

	delimeter = ' '

# Initialize adjacency matrix
adj_matrix = np.zeros((dim,dim))

# Set of graphs were edge weight 1
force_one = {'facebook'}

# Create adjacency matrix
for edge in data:
	# Split entry
	row_data = edge.split(delimeter)

	# Set source
	source = int(row_data[0]) - 1

	# Set sink
	sink = int(row_data[1]) - 1

	# Force weight to one for specified graphs
	if sys.argv[1] in force_one:
		adj_matrix[source, sink] = 1
		continue

	# Skip bad row
	if len(row_data) < 3:
		continue

	# Set weight
	if 'e' not in row_data[2]:
		weight = float(row_data[2])
	else:
		num = row_data[2].split('e')
		exp = int(num[1][1:])
		weight = int(num[0]) * 10**exp


	# Update adjacency
	adj_matrix[source, sink] = weight

# Create sparse matrix
adj_matrix = sparse.csc_matrix(adj_matrix)

# Save sparse matrix
sparse.save_npz('processed/' + sys.argv[1] + '.npz', adj_matrix)
