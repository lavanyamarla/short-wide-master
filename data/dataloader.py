import sys
import numpy as np
from scipy import sparse

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

# Create adjacency matrix
for edge in data:
	# Split entry
	source = edge.split(delimeter)

	# Skip bad row
	if len(source) < 3:
		continue

	# Set sink
	sink = int(source[1]) - 1

	# Set weight
	if 'e' not in source[2]:
		weight = float(source[2])
	else:
		num = source[2].split('e')
		exp = int(num[1][1:])
		weight = int(num[0]) * 10**exp

	# Set source
	source = int(source[0]) - 1

	# Update adjacency
	adj_matrix[source, sink] = weight

# Create sparse matrix
adj_matrix = sparse.csc_matrix(adj_matrix)

# Save sparse matrix
sparse.save_npz('processed/' + sys.argv[1] + '.npz', adj_matrix)
