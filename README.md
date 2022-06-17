# Total Variation Primal Dual Approaches to Laplacian and Poisson Learning

## Senior Honors Thesis Project

### Author: Atef Ali

The code contained in this repository is for my senior honors thesis project, which aims to solve the Poisson and Laplacian Learning problems using an alternative method of gradient descent
that utilizes a primal dual approach to solving the energy minimization problems in each respective scenario.

There are three separate tasks considered in this project, which are listed below in terms of "difficulty to learn"

1. Binary classification of the Two Moons dataset
2. Binary classification of the MNIST dataset (only digits 0 and 1)
3. Multiclass classification of the MNIST dataset (all digits 0-9)


In order to run the code in this project, you need a few packages, namely the following:

* `graphlearning`
* `scikit-learn`
* `numpy`
* `pandas`
* `matplotlib`
* `tqdm`


I would recommend creating a conda environment from the `environment.yml` file included.


## Files
There are 6 files that can be run for various simulations/datasets.
Running any of them will run a given simulation setup with user-configurable experimental hyperparameters. The hyperparameters can be found in the `__main__` section of each file. Each file implements a Total Variation Primal Dual gradient descent scheme for its respective energy minimization problem as described in my honors thesis.

For the two moons dataset, there are two main files that can be run. Both files are for training a binary classifier.
* `two_moons_ll.py`: TV-PD approach to Laplacian learning on Two Moons
* `two_moons_pl.py`: TV-PD approach to Poisson Learing on Two Moons

For the MNIST dataset, there are four main files that can be run. Running any of the files requires a download of the MNIST dataset as downloaded by the graphlearning package - this will automatically be downloaded to the location of where the code is in a directory called `./data` if it isn't already downloaded.

The following are for binary classification of digits 0 and 1 in MNIST.
* `mnist_binary_ll.py`: TV-PD approach to Laplacian Learning on MNIST 0 and 1 digits
* `mnist_binary_pl.py`: TV-PD approach to Poisson Learning on MNIST 0 and 1 digits

The following are for multiclass classification of all digits in MNIST, trained using a one-versus-rest approach.
* `mnist_multiclass_ll.py`: TV-PD approach to Laplacian Learning on all MNIST digits
* `mnist_multiclass_pl.py`: TV-PD approach to Poisson Learning on all MNIST digits

The file `metrics.py` contains some utility/accuracy metric calculation functions.

## Running the simulations

To run a simulation for a given problem, simply modify the hyperparameters as desired in the respective file and specify a path for a results csv file to be saved in the `__main__` area, and call 

`python <path-to-file/file.py>`

e.g. `python /place_I_store_python_code/two_moons_ll.py`.

As the simulation is run, some energy and accuracy metrics will be printed to your terminal so you can monitor progress of training. After the simulation finishes, a csv file will be saved at your specified location with hyperparameter values and accuracy metrics for individual trials of your simulation.

## Notes
Although I've finished writing up my honors thesis and graduated, there's still some places that this repository can be taken! If I had more time, I would like to have done a more thorough job searching for hyperparameters or used an intelligent hyperparameter optimization scheme - I basically handpicked/tried a few configurations of hyperparameter settings in order to find values that converged to relatively "high" accuracies.

Feel free to contact me at alixx830 at umn dot edu with any questions/thoughts!

Special thanks for Professor Jeff Calder for advising me throughout the project, and to my committee members Professor Li Wang and Professor Bernardo Cockburn.

The repository of the graphlearning package, developed by Professor Calder, can be found at

https://github.com/jwcalder/GraphLearning

with more detailed documentation at 

https://jwcalder.github.io/GraphLearning/



