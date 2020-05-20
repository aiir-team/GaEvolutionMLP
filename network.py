#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:30, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

import random
import logging


class Network:
    """Represent a network and let us operate on it. Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = None
        self.nn_param_choices = nn_param_choices
        self.paras = {}       # (dict): represents MLP network parameters
        self.create_random()

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.paras[key] = random.choice(self.nn_param_choices[key])

    def set_paras(self, paras):
        self.paras = paras

    def print_network(self):
        """Print out a network."""
        logging.info(self.paras)
        logging.info("Network accuracy: %.3f%%" % (self.accuracy * 100))
