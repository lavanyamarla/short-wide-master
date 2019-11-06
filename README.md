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
Run this file to generate distances for specific dataset

`python run.py [dataset] [weighted] [geo] [short_wide] [method]`

dataset is name of dataset, weighted/geo/short_wide are bin sizes, and method is `d` for Dijkstras or `f` for Floyd-Warshal.

This file can also be used for debugging

`python run.py debug`

### deg.py
### erdos.py
### replot.py
### dataloader.py
