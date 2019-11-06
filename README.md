# Short Wide Code
## Directory Structure
    .
    ├── data                    # Folder with data
    │   ├── dataloader.py       # Script process data
    │   ├── processed           # Processed data (npz file)
    │   |   └── ...
    │   └── raw                 # Raw dataset
    │       └── ...
    ├── plots                   # Folder with plots
    │   └── ... 
    ├── raw_output              # Folder with distances saved (pickle file)
    │   └── ... 
    ├── algo.py                 # Graph class implementation
    ├── connected.py            # Find largest connected component
    ├── deg.py                  # Random graph with same degree list
    ├── erdos.py                # Random Erdős–Rényi graph
    ├── output.txt              # Summary of fit
    ├── README.md               # README
    ├── replot.py               # Code to replot
    └── run.py                  # Code to generate distances for dataset

## Running Code
### run.py
Run this file to generate distances for specific dataset.

`python run.py [dataset] [weighted] [geo] [short_wide] [method]`

dataset is name of dataset, weighted/geo/short_wide are bin sizes, and method is `d` for Dijkstras or `f` for Floyd-Warshal.

This file can also be used for debugging.

`python run.py debug`

### deg.py
Run this file to generate random degree matched graphs.

`python deg.py [weighted] [geo] [short_wide]`

weighted/geo/short_wide are bin sizes.

### erdos.py
Run this file to generate random Erdős–Rényi graphs.

`python erdos.py [p] [dim] [weighted] [geo] [short_wide]`

p is the probability for graph generation, dim is the size of the graph, weighted/geo/short_wide are bin sizes.

### replot.py
Run this file to generate new plots based on existing data.

`python replot.py [dataset] [weighted] [geo] [short_wide] [method]`

dataset is name of dataset, weighted/geo/short_wide are bin sizes, and method is `d` for Dijkstras or `f` for Floyd-Warshal.

### dataloader.py
Run this file to generate processed data in npz format

`python dataloader.py [dataset]`

This function will need to be modified for new datasets.
