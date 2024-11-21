import random
import copy
import matplotlib.pyplot as plt
import numpy as np

# Constants
N = 50            # Length of genome (number of genes)
P = 50            # Population size
MUTRATE = 0.1     # Mutation probability
MUTSTEP = 0.1     # Maximum step size for mutation
MIN = 0.0         # Minimum value for genes
MAX = 1.0         # Maximum value for genes
GENERATIONS = 500  # Number of generations

# Define individual
class Individual:
    def _init_(self):
        self.gene = [0.0] * N  # Gene represented by a list of real numbers
        self.fitness = 0.0      # Fitness initialized to 0

# Test function - Sum of genes
def test_function(ind):
    ind.fitness = sum(ind.gene)  # Fitness is the sum of the real-valued genes
    return ind.fitness

# Rastrigin function for more complex minimization
def rastrigin_function(ind):
    A = 10
    ind.fitness = A * N + sum(x**2 - A * np.cos(2 * np.pi * x) for x in ind.gene)
    return ind.fitness

# Initialize population randomly
def initialize_population():
    population = []
    for _ in range(P):
        new_ind = Individual()
        new_ind.gene = [random.uniform(MIN, MAX) for _ in range(N)]  # Randomly generate real numbers
        population.append(new_ind)
    return population

# Tournament selection
def tournament_selection(population):
    offspring = []
    for _ in range(P):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        offspring.append(parent1 if parent1.fitness < parent2.fitness else parent2)
    return offspring

# Single-point crossover
def crossover(offspring):
    for i in range(0, P, 2):
        if i + 1 < P:
            toff1 = copy.deepcopy(offspring[i])
            toff2 = copy.deepcopy(offspring[i + 1])
            crosspoint = random.randint(1, N - 1)
            toff1.gene[crosspoint:], toff2.gene[crosspoint:] = toff2.gene[crosspoint:], toff1.gene[crosspoint:]
            offspring[i], offspring[i + 1] = toff1, toff2
    return offspring

# Real-valued mutation
def mutate(offspring):
    for ind in offspring:
        for j in range(N):
            if random.random() < MUTRATE:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene = ind.gene[j] + alter
                gene = max(MIN, min(gene, MAX))
                ind.gene[j] = gene
    return offspring

# Evaluate the population
def evaluate_population(population):
    for ind in population:
        test_function(ind)  # Replace with rastrigin_function(ind) for the Rastrigin function

# Genetic algorithm main loop with cloud-like 3D visualization
def genetic_algorithm():
    population = initialize_population()
    all_generations = []  # Stores all fitness values for plotting
    
    for g in range(GENERATIONS):
        offspring = copy.deepcopy(population)
        evaluate_population(offspring)
        
        # Collect fitness values for this generation
        generation_data = [(g, ind.fitness) for ind in offspring]
        all_generations.extend(generation_data)

        offspring = tournament_selection(offspring)
        offspring = crossover(offspring)
        offspring = mutate(offspring)
        
        population = copy.deepcopy(offspring)

    # Split data for 3D scatter plot
    generations = [gen[0] for gen in all_generations]
    fitness_values = [gen[1] for gen in all_generations]
    individuals = range(len(all_generations))

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(generations, individuals, fitness_values, c=fitness_values, cmap='viridis', marker='o', alpha=0.6)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Individual')
    ax.set_zlabel('Fitness Value')
    plt.title('Genetic Algorithm: Fitness Distribution Over Generations')
    fig.colorbar(scatter, ax=ax, label='Fitness Value')
    plt.show()

# Run the genetic algorithm
genetic_algorithm()