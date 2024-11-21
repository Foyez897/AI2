import random
import copy

# Define the Individual class representing a candidate solution
class Individual:
    def __init__(self, N):
        self.gene = [0] * N  # Binary gene array (0s and 1s)
        self.fitness = 0      # Fitness value of the individual

# Define the test function (fitness function) that counts the number of '1's in the gene
def test_function(ind):
    utility = sum(ind.gene)  # Fitness is the number of '1's in the gene array
    return utility

# Function to initialize the population with P individuals, each with N genes
def initialize_population(P, N):
    population = []
    for _ in range(P):
        tempgene = [random.randint(0, 1) for _ in range(N)]  # Random binary genes (0 or 1)
        new_ind = Individual(N)                              # Create a new individual
        new_ind.gene = tempgene.copy()                       # Set the individual's genes
        new_ind.fitness = test_function(new_ind)             # Evaluate the individual's fitness
        population.append(new_ind)                           # Add the individual to the population
    return population

# Tournament selection function to select the fittest individuals and create offspring
def tournament_selection(population, P, T):
    offspring = []
    for _ in range(P):
        # Randomly select T individuals from the population
        selected = random.sample(population, T)
        # Choose the individual with the highest fitness from the selected group
        fittest = max(selected, key=lambda ind: ind.fitness)
        offspring.append(copy.deepcopy(fittest))  # Make a copy of the fittest individual for the offspring
    return offspring

# Parameters
N = 10  # Number of genes in each individual
P = 50  # Population size (number of individuals)
T = 2   # Tournament size (number of individuals selected for tournament)

# Initialize and evaluate the population
population = initialize_population(P, N)

# Perform tournament selection to generate the offspring population
offspring_population = tournament_selection(population, P, T)

# Calculate and compare the total fitness of the original and offspring populations
total_fitness_original = sum(ind.fitness for ind in population)
total_fitness_offspring = sum(ind.fitness for ind in offspring_population)

# Print the total fitness of both populations
print("Total fitness of original population:", total_fitness_original)
print("Total fitness of offspring population:", total_fitness_offspring)
