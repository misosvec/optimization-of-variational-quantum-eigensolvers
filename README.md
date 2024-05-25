# Optimization of variational quantum eigensolvers

All source files for this bachelor's thesis are publicly available in this repository.

Files [ansatzes.py](src/ansatzes.py) and [optimizers.py](src/optimizers.py) handle the initialization of ansatzes and
optimizers. File [benchmark.py](src/benchmark.py) runs the VQE in a multiprocessing manner and saves
produced data to a CSV file. Juptyer notebook [visualizations.ipynb](src/visualizations.ipynb) contains all
visualizations of circuits, qubits, and gates used in our thesis. Plots representing results
are located in file [results.ipynb](src/results.ipynb).

To run the benchmarking, use the command `python3 benchmark.py`. Before doing so, it is necessary to install dependencies from the [requirements.txt](src/requirements.txt) file by running `pip install -r requirements.txt`. This will generate a `data.csv` file that can be used for further visualizations.