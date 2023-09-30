# Towards Training Without Depth Limits: Large Batch Normalization Without Gradient Explosion

Code for the paper: Towards Training Without Depth Limits: Large Batch Normalization Without Gradient Explosion.

The repository contains all the necessary code to reproduce the experiments inside of the `src/` directory:

- `modules`: contains the code defining the neural network modules (`models.py`), as well as various utility functions for training, testing, data loading and measurements (`data_utils.py`, `utils.py`)
- `theorem_validations`: contains Jupyter notebooks for reproducing the results from the main figures in the paper i.e. the main theorems
- `environment.yml`: the conda environment containing the necessary packages for reproducing the results; install using `conda env create -f environment.yml`

For more information about each concept, see the paper content.

In order to reproduce the training accuracy plots, as well as the implicity SGD orthogonality results, run `run.py` on a dataset of your choice, and then plot the resulting `.csv` files containing the results. See `run.py` for the main command line arguments required to execute the script.

---

Author contact: [Alexandru Meterez](mailto:alexandrumeterez@gmail.com).
