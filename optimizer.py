#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:30, 20/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from functools import reduce
from operator import add
import random
import logging
from copy import deepcopy
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
            nn_param_choices (dict): Possible network paremters
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

    def create_solution(self):
        return Network(self.nn_param_choices)

    @staticmethod
    def calculate_fitness(network, dataset):
        """Return the accuracy, which is our fitness function."""
        train_and_score(network, dataset)
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice([mother.network[param], father.network[param]])

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def create_new_population(self):
        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

    def evolve(self, dataset):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        networks = [self.create_solution() for _ in range(self.pop_size)]
        # Get scores for each network.
        pop = [(network, self.calculate_fitness(network, dataset)) for network in networks]
        # Sort on the scores.
        networks = [item[0] for item in sorted(pop, key=lambda x: x[0], reverse=True)]          # Higher is better because fitness is accuracy
        # Get the g_best
        g_best = deepcopy(networks[0])

        # Evolve the generation.
        for epoch in range(self.max_gens):
            logging.info("***Doing generation %d of %d***" % (epoch + 1, self.max_gens))

            # Print out the average accuracy each generation.
            logging.info("\tBest accuracy: %.2f%%" % (average_accuracy * 100))
            logging.info('-' * 80)

            # Create new population
                networks = optimizer.evolve(networks)

        # Sort our final population.
        networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

        # Print out the top 5 networks.
        print_networks(networks[:5])


        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]


