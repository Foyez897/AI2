import random
import copy
import matplotlib.pyplot as plt

# Define the Individual class representing a single candidate solution
class Individual:
    def __init__(self, N):  # Constructor to initialize an individual with N binary genes
        self.gene = [0] * N  # Genes are binary values (0 or 1)
        self.fitness = 0  # Fitness value, initially set to 0

# Function to evaluate fitness (Counting Ones problem)
def evaluate_fitness(ind):
    return sum(ind.gene)  # Fitness is the number of 1s in the gene list

# Initialize the population with P individuals, each having N genes
def initialize_population(P, N):  # Create the initial population
    population = []
    for _ in range(P):
        new_ind = Individual(N)  # Create a new individual
        new_ind.gene = [random.randint(0, 1) for _ in range(N)]  # Initialize genes with random binary values
        new_ind.fitness = evaluate_fitness(new_ind)  # Calculate initial fitness
        population.append(new_ind)  # Add individual to the population
    return population

# Tournament selection function to choose the fittest individuals for reproduction
def tournament_selection(population, T):
    offspring = []
    P = len(population)
    for _ in range(P):
        candidates = random.sample(population, T)  # Randomly select T candidates from the population
        fittest = max(candidates, key=lambda ind: ind.fitness)  # Choose the candidate with the highest fitness
        offspring.append(copy.deepcopy(fittest))  # Make a copy of the fittest candidate for the next generation
    return offspring

# Single-point crossover function to combine the genes of two parents
def crossover(parent1, parent2, N):
    crosspoint = random.randint(1, N-1)  # Choose a random crossover point
    child1 = copy.deepcopy(parent1)  # Create copies of the parents for the children
    child2 = copy.deepcopy(parent2)
    # Swap genes after the crossover point between parent1 and parent2
    child1.gene[crosspoint:], child2.gene[crosspoint:] = parent2.gene[crosspoint:], parent1.gene[crosspoint:]
    return child1, child2

# Mutation function to randomly flip bits in an individual's genes
def mutate(individual, mutation_rate):
    for i in range(len(individual.gene)):
        if random.random() < mutation_rate:  # With a probability of mutation_rate, flip the bit
            individual.gene[i] = 1 - individual.gene[i]  # Flip the bit (0 becomes 1, 1 becomes 0)

# Main genetic algorithm function to evolve the population
def genetic_algorithm(P, N, T, generations, crossover_rate, mutation_rate):
    population = initialize_population(P, N)  # Create the initial population
    avg_fitness_history = []  # List to track average fitness over generations
    best_fitness_history = []  # List to track best fitness over generations

    for gen in range(generations):
        print(f"Generation {gen + 1}")

        # Select offspring using tournament selection
        offspring_population = tournament_selection(population, T)

        # Apply crossover to create new offspring
        for i in range(0, P, 2):
            if random.random() < crossover_rate and i + 1 < P:  # Apply crossover with probability crossover_rate
                parent1, parent2 = offspring_population[i], offspring_population[i + 1]
                child1, child2 = crossover(parent1, parent2, N)  # Perform crossover
                offspring_population[i] = child1
                offspring_population[i + 1] = child2

        # Apply mutation to the offspring population
        for ind in offspring_population:
            mutate(ind, mutation_rate)  # Apply mutation to each individual
            ind.fitness = evaluate_fitness(ind)  # Recalculate fitness after mutation

        # Replace the old population with the new offspring population
        population = offspring_population

        # Calculate and store fitness statistics
        avg_fitness = sum(ind.fitness for ind in population) / P  # Calculate average fitness
        best_fitness = max(ind.fitness for ind in population)  # Find the best fitness in the population
        avg_fitness_history.append(avg_fitness)  # Store average fitness for this generation
        best_fitness_history.append(best_fitness)  # Store best fitness for this generation

        print(f"Average fitness: {avg_fitness:.2f}, Best fitness: {best_fitness}")

    return avg_fitness_history, best_fitness_history

# Parameters for the genetic algorithm
P = 50  # Population size
N = 50  # Number of genes (length of the binary string)
T = 3  # Tournament size for selection
generations = 50  # Number of generations to run the algorithm
crossover_rate = 0.7  # Probability of crossover between parents
mutation_rate = 0.01  # Probability of mutation for each gene

# Run the genetic algorithm and collect fitness history
avg_fitness_history, best_fitness_history = genetic_algorithm(P, N, T, generations, crossover_rate, mutation_rate)

# Plot the average and best fitness over generations
plt.figure(figsize=(10, 6))
plt.plot(range(1, generations + 1), avg_fitness_history, label='Average Fitness', marker='o')
plt.plot(range(1, generations + 1), best_fitness_history, label='Best Fitness', marker='x')
plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.xticks(range(1, generations + 1, 5))  # Set x-axis ticks every 5 generations
plt.grid()
plt.legend()
plt.show()
