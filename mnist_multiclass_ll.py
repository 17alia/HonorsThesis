import numpy as np
import graphlearning as gl
from metrics import calculate_energy_ll, calculate_onevrest_acc
import pandas as pd

"""
This file contains code for running trials of total variation primal dual laplacian learning
using the mnist dataset with all digits 0-9.
The code under __main__ can be configured based on desired experimental setup.
"""

def single_trial(X, labels, label_rate, graph, alpha, lmda, max_iters=10000, tol=1e-8):
    """runs a single trial of training total variation laplacian learning on the mnist dataset with all 10 digits 0-9.
       uses a one-vs-rest approach to binary classify all 10 digits, and computes a class confidence score.
    Args:
        X (np.ndarray): array of datapoints
        labels (np.ndarray): array of labels
        label_rate (int): number of training labels to use per class
        graph (graphlearning.graph): graph encoding relations between datapoints
        alpha (float): learning rate hyperparameter
        lmda (float): regularization hyperparameter
        max_iters (int, optional): maximum number of iterations to train for. Defaults to 10000.
        tol (float, optional): tolerance parameter. every 100 iterations, we check if energy changes by more than this, and if it does, we stop training.. Defaults to 1e-8.

    Returns:
        label_rate (int): number of training labels per class
        alpha (float): learning rate hyperparamter
        lmda (float): regularization hyperparameter
        final_accuracy (float): accuracy at end of training 
    """
    train_ind = gl.trainsets.generate(labels, rate=label_rate) # get indices of training labels at given label rate

    n = X.shape[0]

    per_class_u = {} # to store per class confidence scores

    W_reweighted = graph.reweight(train_ind)
    graph = gl.graph(W_reweighted)

    for positive_label in range(10): # iterate over digits 0-9 using a one-v-rest binary classifcation approach
        print(f"Positive class: {positive_label}")
        zero_one_labels = labels.copy()
        zero_one_labels[labels != positive_label] = 0
        zero_one_labels[labels == positive_label] = 1

        u = np.random.uniform(-1, 1, size=(n,)) # vector used to help make label decisions
        f = np.zeros((n,))
        xi = graph.gradient(u=u, weighted=False) # vector field used to help make label decisions
        f[train_ind[zero_one_labels[train_ind] == 1]] = 1 # f is a vector with 1 in positive class, -1 in negative class, and 0s in unlabeled points
        f[train_ind[zero_one_labels[train_ind] == 0]] = -1

        label_decisions = np.zeros((n,)) # vector to contain final label decisions

        i = 0
        energy_list = [] # used to check energy every 1000 iterations
        energy = calculate_energy_ll(lmda, graph, xi, f) + 1 # initialize energy for stopping conditions
        energy_list.append(energy)
        while i < max_iters:
            numerator = xi + (alpha * graph.gradient(graph.divergence(xi, weighted=True) + ((lmda ** -1) * f), weighted=False))
            denominator = alpha * np.absolute(graph.gradient(graph.divergence(xi, weighted=True) + ((lmda ** -1) * f), weighted=False))
            denominator.data = denominator.data + 1
            denominator.data = 1 / denominator.data
            xi = numerator.multiply(denominator)
            u = f + lmda * graph.divergence(xi, weighted=True)
            label_decisions[u < 0] = 0 # get label decisions at iteration i based on computed u
            label_decisions[u > 0] = 1

            if i % 100 == 0 and i != 0: # print accuracy and energy at every 100th iteration
                energy = calculate_energy_ll(lmda, graph, xi, f)
                
                if abs(energy - energy_list[-1]) / energy_list[-1] < tol: # if change in energy is less than tolerance param, stop training
                    break
                energy_list.append(energy)
                print(f"Digit {positive_label} energy at iteration {i}: {energy}")
                print(f"Digit {positive_label} accuracy at iteration {i}: {calculate_onevrest_acc(label_decisions, zero_one_labels)}")
            i += 1
        u = f + lmda * graph.divergence(xi, weighted=True)

        per_class_u[positive_label] = u

    # loop over all per class confidence scores to find final labels
    predicted_labels = np.zeros((n,))
    for idx in range(n):
        current_max_value = -1
        for key in per_class_u.keys():
            if per_class_u[key][idx] > current_max_value:
                current_max_value = per_class_u[key][idx]
                predicted_labels[idx] = key
    
    final_accuracy = gl.ssl.ssl_accuracy(predicted_labels, labels, len(train_ind))
    print(f"\nAccuracy over all classes is: {final_accuracy}")

    return label_rate, alpha, lmda, final_accuracy


if __name__ == "__main__":

    X, labels = gl.datasets.load('mnist', labels_only=False, metric='vae')
    alphas = {1: 15.0, 2: 9.6, 3: 5.4, 4: 2.4, 5: 0.6} # it seems like adjusting alpha param based on number of labeled points leads to better accuracy/convergence
    lmda = 0.6
    k = 10 # number of nearest neighbors for knn graph
    W_unweighted = gl.weightmatrix.knn(X, k, metric='vae')
    graph = gl.graph(W_unweighted)
    num_trials = 10 # number of trials per label rate
    csv_output_path = f'./results/mnistmulti_ll_numtrials{num_trials}.csv'

    df_accuracies = pd.DataFrame(columns=['label_rate', 'alpha', 'lambda', 'accuracy'])

    for trial_num in range(num_trials):
        print(f"Trial number {trial_num}")

        for label_rate in range(1, 6): # we test algorithsm from label rates of 1 to 5 labels per class
            print(f"Label rate: {label_rate}")
            lr, a, l, acc = single_trial(X, labels, label_rate, graph, alphas[label_rate], lmda)
            df_accuracies = df_accuracies.append({'label_rate': lr, 'alpha': a, 'lambda': l, 'accuracy': acc})
    df_accuracies.to_csv(csv_output_path)