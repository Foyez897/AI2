import random
import copy
import matplotlib.pyplot as plt
import numpy as np

# Constants
d = 20            # Number of genes
P = 300            # Population size
MUTRATE = 0.1      # Mutation probability
MUTSTEP = 25.0     # Maximum step size for mutation
MIN = -500.0       # Minimum value for genes
MAX = 500.0        # Maximum value for genes
GENERATIONS = 50   # Number of generations
YOU = 30           # Replace with the last two digits of your student number

# Define individual
class Individual:
    def __init__(self):
        self.gene = [0.0] * d  # Gene represented by a list of real numbers
        self.fitness = 0.0     # Fitness initialized to 0

# Modified fitness function
def custom_function(ind):
    """
    f(x) = 418.9829d - sum(x_i * sin(sqrt(|x_i|))) + 30
    """
    xi = np.array(ind.gene)
    ind.fitness = 418.9829 * d - np.sum(xi * np.sin(np.sqrt(np.abs(xi)))) + 30
    return ind.fitness


# Initialize population randomly
def initialize_population():
    population = []
    for _ in range(P):
        new_ind = Individual()
        new_ind.gene = [random.uniform(MIN, MAX) for _ in range(d)]  # Randomly generate real numbers
        population.append(new_ind)
    return population

# Tournament selection
def tournament_selection(population):
    offspring = []
    for _ in range(P):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        offspring.append(parent1 if parent1.fitness < parent2.fitness else parent2)  # Minimization
    return offspring

# Single-point crossover
def crossover(offspring):
    for i in range(0, P, 2):
        if i + 1 < P:
            toff1 = copy.deepcopy(offspring[i])
            toff2 = copy.deepcopy(offspring[i + 1])
            crosspoint = random.randint(1, d - 1)
            toff1.gene[crosspoint:], toff2.gene[crosspoint:] = toff2.gene[crosspoint:], toff1.gene[crosspoint:]
            offspring[i], offspring[i + 1] = toff1, toff2
    return offspring

# Real-valued mutation
def mutate(offspring):
    for ind in offspring:
        for j in range(d):
            if random.random() < MUTRATE:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene = ind.gene[j] + alter
                ind.gene[j] = max(MIN, min(MAX, gene))  # Ensure bounds
    return offspring

# Evaluate the population
def evaluate_population(population):
    for ind in population:
        custom_function(ind)

# Genetic algorithm main loop
def genetic_algorithm():
    population = initialize_population()
    avg_fitness_history = []
    best_fitness_history = []

    for g in range(GENERATIONS):
        evaluate_population(population)
        avg_fitness = np.mean([ind.fitness for ind in population])
        best_fitness = min([ind.fitness for ind in population])  # Minimization

        avg_fitness_history.append(avg_fitness)
        best_fitness_history.append(best_fitness)

        print(f"Generation {g + 1}: Average Fitness = {avg_fitness:.2f}, Best Fitness = {best_fitness:.2f}")

        offspring = tournament_selection(population)
        offspring = crossover(offspring)
        offspring = mutate(offspring)
        population = copy.deepcopy(offspring)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, GENERATIONS + 1), avg_fitness_history, label='Average Fitness', marker='o')
    plt.plot(range(1, GENERATIONS + 1), best_fitness_history, label='Best Fitness', marker='x')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.show()

# Run the genetic algorithm
genetic_algorithm()
