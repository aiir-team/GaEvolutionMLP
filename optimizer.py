#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:30, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy.random import choice, randint, uniform
from copy import deepcopy
import logging
from network import Network
from network_helper import train_and_score

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


class GaOptimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, max_gens=10, pop_size=10, retain=0.4, random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network parameters
            retain (float): Percentage of population to retain after each generation
            random_select (float): Probability of a rejected network remaining in the population
            mutate_chance (float): Probability a network will be randomly mutated
        """
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices
        self.model = None

    def create_solution(self):
        return Network(self.nn_param_choices)

    @staticmethod
    def calculate_fitness(network, dataset):
        """Return the accuracy, which is our fitness function."""
        acc = train_and_score(network.paras, dataset)
        network.__setattr__("accuracy", acc)
        return acc

    def __breed__(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (Network): Network parameters
            father (Network): Network parameters
        Returns:
            (list): Two network objects
        """
        n_paras = len(mother.paras.keys())
        cut_point = randint(1, n_paras-1)

        child1, child2 = {}, {}
        # Loop through the parameters and pick params for the kid.
        for idx, param in enumerate(self.nn_param_choices):
            if idx < cut_point:
                child1[param] = mother.paras[param]
                child2[param] = father.paras[param]
            else:
                child1[param] = father.paras[param]
                child2[param] = mother.paras[param]

        # Now create a network object.
        network1 = Network(self.nn_param_choices)
        network1.set_paras(child1)
        network2 = Network(self.nn_param_choices)
        network2.set_paras(child2)

        # Randomly mutate some of the children.
        if self.mutate_chance > uniform():
            network1 = self.__mutate__(network1)
        if self.mutate_chance > uniform():
            network2 = self.__mutate__(network2)
        return [network1, network2]

    def __mutate__(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (Network): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """
        # Choose a random key.
        mutation = choice(list(self.nn_param_choices.keys()))
        # Mutate one of the params. Make sure the mutated is different
        network.paras[mutation] = choice(list(set(self.nn_param_choices[mutation]) - {network.paras[mutation]}))
        return network

    def create_new_population(self, pop):
        """ Remember pop already sorted by fitness """

        retain_length = int(self.pop_size * self.retain)    # Get the number we want to keep for the next gen.
        parents = pop[:retain_length]                       # The parents are every network we want to keep.

        for individual in pop[retain_length:]:              # For those we aren't keeping, randomly keep some anyway.
            if self.random_select > uniform():
                parents.append(individual)

        parents_length = len(parents)                       # Now find out how many spots we have left to fill.
        desired_length = self.pop_size - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad and they aren't the same network...
            male, female = choice(range(0, parents_length), 2, replace=False)
            male, female = parents[male], parents[female]

            babies = self.__breed__(male, female)           # Breed them.
            for baby in babies:                             # Add the children one at a time.
                if len(children) < desired_length:          # Don't grow larger than desired length.
                    children.append(baby)
        parents.extend(children)
        return parents

    def evolve(self, dataset):
        """Evolve a population of networks.
        Args:
            dataset ():
            pop (list): A list of network parameters
        Returns:
            (Network): The evolved population of networks
        """
        networks = [self.create_solution() for _ in range(self.pop_size)]
        # Get scores for each network.
        pop = [(net, self.calculate_fitness(net, dataset)) for net in networks]
        # Sort on the scores. Higher is better because fitness is accuracy
        networks = [item[0] for item in sorted(pop, key=lambda x: x[1], reverse=True)]
        self.model = deepcopy(networks[0])  # Get the g_best

        # Evolve the generation.
        for epoch in range(self.max_gens):
            # Create new population
            networks = self.create_new_population(networks)
            # Get scores for each network.
            pop = [(network, self.calculate_fitness(network, dataset)) for network in networks]
            # Sort on the scores. Higher is better because fitness is accuracy
            networks = [item[0] for item in sorted(pop, key=lambda x: x[1], reverse=True)]

            if self.model.accuracy > networks[0].accuracy:
                self.model = deepcopy(networks[0])
            # Print out the best accuracy each generation.
            logging.info("Epoch: %d/%d, best accuracy: %.4f%%" % (epoch+1, self.max_gens, self.model.accuracy * 100))

        return self.model
