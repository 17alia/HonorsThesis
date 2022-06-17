import sklearn.datasets as datasets
import graphlearning as gl
import pandas as pd
import numpy as np
from metrics import calculate_energy_ll
import os

"""
This file contains code for running trials of total variation primal dual laplacian learning
in the binary classification setting for MNIST with digits 0 and 1.
The code under __main__ can be configured based on desired experimental setup.
"""

def single_trial(X, labels, label_rate, graph, alpha, lmda, max_iters=10000, tol=1e-8):
    """runs a single trial of training total variation laplacian learning on the mnist dataset to binary classify digits 0 and 1.

    Args:
        X (np.ndarray): array of datapoints
        labels (np.ndarray): array of labels
        label_rate (int): number of training labels to use per class
        graph (graphlearning.graph): graph encoding relations between datapoints
        alpha (float): learning rate hyperparameter
        lmda (float): regularization hyperparameter
        max_iters (int, optional): maximum number of iterations to train for. Defaults to 10000.
        tol (float, optional): tolerance parameter. every 1000 iterations, we check if energy changes by more than this, and if it does, we stop training.. Defaults to 1e-8.

    Returns:
        label_rate (int): number of training labels per class
        alpha (float): learning rate hyperparamter
        lmda (float): regularization hyperparameter
        final_accuracy (float): accuracy at end of training 
    """
    train_ind = gl.trainsets.generate(labels, rate=label_rate) # get indices of training labels at given label rate

    n = X.shape[0]

    u = np.random.uniform(-1, 1, size=(n,)) # vector used to help make label decisions
    f = np.zeros((n,))
    xi = graph.gradient(u=u, weighted=False) # vector field used to help make label decisions
    f[train_ind[labels[train_ind] == 1]] = 1 # f is a vector with 1 in positive class, -1 in negative class, and 0s in unlabeled points
    f[train_ind[labels[train_ind] == 0]] = -1
    
    energy = calculate_energy_ll(lmda, graph, xi, f) + 1 # initialize energy for stopping conditions
    label_decisions = np.zeros((n,)) # vector to contain final label decisions

    initial_label_decisions = np.zeros((n,)) # vector to calculate initial accuracy/energy
    initial_label_decisions[u < 0] = 0
    initial_label_decisions[u >= 0] = 1

    print(f"Initial energy: {calculate_energy_ll(lmda, graph, xi, f)}")
    print(f"Initial accuracy: {gl.ssl.ssl_accuracy(initial_label_decisions, labels, num_train=len(train_ind))}")


    i = 0
    energy_list = [] # used to check energy every 1000 iterations
    energy_list.append(calculate_energy_ll(lmda, graph, xi, f)+1)
    while i < max_iters:
        numerator = xi + (alpha * graph.gradient(graph.divergence(xi, weighted=True) + ((lmda ** -1) * f), weighted=False))
        denominator = alpha * np.absolute(graph.gradient(graph.divergence(xi, weighted=True) + ((lmda ** -1) * f), weighted=False))
        denominator.data = denominator.data + 1
        denominator.data = 1 / denominator.data
        xi = numerator.multiply(denominator)
        u = f + lmda * graph.divergence(xi, weighted=True)
        label_decisions[u < 0] = 0 # get label decisions at iteration i based on computed u
        label_decisions[u > 0] = 1

        if i % 1000 == 0 and i != 0: # print accuracy and energy at every 1000th iteration
            energy = calculate_energy_ll(lmda, graph, xi, f)
            
            if abs(energy - energy_list[-1]) / energy_list[-1] < tol: # if energy change is less than tolerance parameter, stop training
                break
            energy_list.append(energy)
            print(f"Energy at iteration {i}: {energy}")
            print(f"Accuracy at iteration {i}: {gl.ssl.ssl_accuracy(label_decisions, labels, num_train=len(train_ind))}")
        
        i += 1
    
    final_accuracy = gl.ssl.ssl_accuracy(label_decisions, labels, num_train=len(train_ind))
    print(f"Final accuracy ({i} iterations): {final_accuracy}")
    print(f"Final Energy ({i} iterations): {calculate_energy_ll(lmda, graph, xi, f)}")

    return label_rate, alpha, lmda, final_accuracy





if __name__ == '__main__':
    X, labels = gl.datasets.load('mnist', labels_only=False, metric='vae')
    X = X[(labels == 0) | (labels == 1)]
    labels = labels[(labels == 0) | (labels == 1)]
    alpha = 0.6 # learning rate hyperparam
    lmda = 0.6 # regularization hyperparam
    k = 10 # number of nearest neighbors in KNN graph
    W_unweighted = gl.weightmatrix.knn(X, k)
    graph = gl.graph(W_unweighted)
    num_trials = 10 # number of trials per label rate
    csv_output_path = f'./results/mnistbinary_ll_numtrials{num_trials}.csv'
    csv_output_dir = os.path.dirname(csv_output_path)
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    df_accuracies = pd.DataFrame(columns=['label_rate', 'alpha', 'lambda', 'accuracy'])

    for trial_num in range(num_trials):
        print(f"Trial number: {trial_num}")

        for label_rate in range(1, 6): # we test algorithms from label rates of 1 to 5 labels per class
            print(f"Label rate: {label_rate}")
            lr, a, l, acc = single_trial(X, labels, label_rate, graph, alpha, lmda, max_iters=10000, tol=1e-5)
            df_accuracies = df_accuracies.append({'label_rate': lr, 'alpha': a, 'lambda': l, 'accuracy': acc}, ignore_index=True)

    df_accuracies.to_csv(csv_output_path)



    


