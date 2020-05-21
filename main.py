#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:30, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%
"""Entry point to evolving the neural network. Start here."""
    # 2520 brute force
    # 20 + 10 * 20 = 220 GA

import logging
from optimizer import GaOptimizer as Optimizer

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

if __name__ == '__main__':
    generations = 20                         # Number of times to evole the population.
    population = 10                          # Number of networks in each generation.
    dataset = 'mnist'                       # cifar10 / mnist

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    logging.info("***Evolving %d generations with population %d***" % (generations, population))
    model = Optimizer(nn_param_choices, generations, population, retain=0.4, random_select=0.1, mutate_chance=0.2)
    best_network = model.evolve(dataset)
    best_network.print_network()
