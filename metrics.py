import numpy as np

"""
This file is for calculating various accuracy and energy metrics, functions as a utilities file.
"""

def calculate_energy_ll(lmda, graph, xi, f):
    """For calculating the energy in laplacian learning problems.
       Energy is given by the expression 0.5 * 2norm(lambda * div(xi) + f) ^ 2

    Args:
        lmda (float): regularization hyperparameter lambda
        graph (graphlearning.graph): graph encoding relations between datapoints
        xi (np.ndarray): vector used to help make label decisions
        f (np.ndarray): array with ground truth labels at given label rate
    """
    return 0.5 * np.linalg.norm(lmda * graph.divergence(xi) + f, ord=2) ** 2

def calculate_energy_pl(lmda, graph, xi, f):
    """For calculating the energy in poisson learning problems.
    Energy is given by the expression 1norm(lambda * div(xi) + f)

    Args:
        lmda (float): regularization hyperparameter lambda
        graph (graphlearning.graph): graph encoding relations between datapoints
        xi (np.ndarray): vector field/matrix used to help make label decisions
        f (np.ndarray): vector with ground truth labels at given label rate
    """
    return np.linalg.norm(lmda * graph.divergence(xi) + f, ord=1)

def calculate_onevrest_acc(label_decisions, gt_labels):
    """for calculating the accuracy as an intermediate check in one-v-rest approach for single positive class vs all negatives

    Args:
        label_decisions (np.ndarray): predicted labels
        gt_labels (np.ndarray): ground truth labels

    Returns:
        float: accuracy of single positive class vs rest of negatives.
    """
    return np.sum(label_decisions == gt_labels) / gt_labels.shape[0] * 100

    