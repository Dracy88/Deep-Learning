import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.RandomForest import RandomForest

import matplotlib.pyplot as plt
from time import time
import numpy as np


''' Settings of the program '''
np.random.seed(7)
budget = 100
n_initial_evaluation = 20
n_estimators = 10 # Number of trees for random forest
optimizer = [1, 3]
n_layers = [0, 2]
neurons_l1 = [128, 4097]
neurons_l2 = [128, 4097]


def objective_function(lr, optimizer, n_layers, neurons_l1, neurons_l2):
    ''' This function return the validation loss with a specific hyper parameters set'''

    print("lr:", lr, "optimizer:", optimizer, "n_layers:", n_layers, "neurons_l1:", neurons_l1, "neurons_l2:", neurons_l2)
    val_acc = ""
    while val_acc is "":
        val_acc = float(input('Enter your val_acc obtained: '))

    return val_acc


def SMBO(model, acquisition):
    ''' Define SMBO function for obtain best hyper-parameter '''

    # Setting the range of the hyper-parameter to test
    param = {'lr': ('cont', [0.00001, 0.1]),
             'optimizer': ('int', optimizer),
             'n_layers': ('int', n_layers),
             'neurons_l1': ('int', neurons_l1),
             'neurons_l2': ('int', neurons_l2)}

    # Start to measure total time needed for the elaboration
    start_time = time()

    # Setting and running GPGO function
    gpgo = GPGO(model, acquisition, objective_function, param, n_jobs=1)
    gpgo.run(max_iter=budget - n_initial_evaluation, init_evals=n_initial_evaluation)

    # Printing the total time required for the elaboration
    print("Total execution time:", int((time() - start_time)), "seconds")

    # Plotting history of the best value seen
    plt.plot(gpgo.history)
    plt.title('Accuracy vs Iterations')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('# Iterations')
    plt.show()

    print("Best set of hyper-parameters found:")
    print(gpgo.getResult())

    return gpgo.history


model = RandomForest(n_estimators=n_estimators)
acquisition = Acquisition(mode='ProbabilityImprovement')
print("Surrogate Model: Random Forest")
print("Acquisition Function: Probability of Improvement")

test_history = SMBO(model, acquisition)
