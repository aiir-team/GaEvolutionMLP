#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:41, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%
"""Iterate over every combination of hyper-parameters."""

import logging
from network import Network
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='brute-log.txt'
)


def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        network.print_network()
        pbar.update(1)
    pbar.close()

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()


def generate_network_list(nn_param_choices):
    """Generate a list of all possible networks.

    Args:
        nn_param_choices (dict): The parameter choices

    Returns:
        networks (list): A list of network objects

    """
    networks = []

    # This is silly.
    for nbn in nn_param_choices['nb_neurons']:
        for nbl in nn_param_choices['nb_layers']:
            for a in nn_param_choices['activation']:
                for o in nn_param_choices['optimizer']:
                    for do in nn_param_choices['dropout']:

                        # Set the parameters.
                        network = {
                            'nb_neurons': nbn,
                            'nb_layers': nbl,
                            'activation': a,
                            'optimizer': o,
                            'dropout': do
                        }
                        # Instantiate a network object with set parameters.
                        network_obj = Network()
                        network_obj.set_paras(network)

                        networks.append(network_obj)
    return networks


if __name__ == '__main__':
    """Brute force test every network."""
    dataset = 'cifar10'

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    logging.info("***Brute forcing networks***")

    networks = generate_network_list(nn_param_choices)

    train_networks(networks, dataset)