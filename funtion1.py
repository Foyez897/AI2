import random
import copy
import matplotlib.pyplot as plt

# Define the Individual class representing a single member of the population
class Individual:
    def __init__(self, N):  # Constructor to initialize an individual with N genes
        self.gene = [0.0] * N  # Initialize all genes with real values (initialized to 0.0)
        self.fitness = 0.0  # Fitness value to be computed later

# Fitness Function 1: Custom fitness function
def fitness_function_1(individual):
    """
    f(x) = (x1 - 1)^2 + Σ (i * (2 * xi^2 - xi-1)^2)
    where -10 <= x <= 10 and d = 20
    """
    x = individual.gene
    d = len(x)  # Dimension of the individual
    fitness = (x[0] - 1) ** 2  # First term (x1 - 1)^2
    for i in range(1, d):  # Iterate from x2 to xd
        fitness += i * (2 * x[i]**2 - x[i - 1])**2
    return fitness

# Adjusted evaluation function for raw fitness
def evaluate_fitness(ind):
    return fitness_function_1(ind)

# Initialize the population with P individuals, each having N real-valued genes
def initialize_population(P, N, MIN, MAX):
    population = []
    for _ in range(P):
        new_ind = Individual(N)  # Create a new individual
        new_ind.gene = [random.uniform(MIN, MAX) for _ in range(N)]  # Assign random real values to genes
        new_ind.fitness = evaluate_fitness(new_ind)  # Calculate the individual's fitness
        population.append(new_ind)  # Add the individual to the population
    return population

# Tournament selection to choose individuals for reproduction
def tournament_selection(population, T):
    offspring = []
    P = len(population)
    for _ in range(P):
        candidates = random.sample(population, T)  # Randomly select T individuals
        fittest = min(candidates, key=lambda ind: ind.fitness)  # Choose the one with the lowest fitness
        offspring.append(copy.deepcopy(fittest))  # Add a copy of the fittest individual to the offspring list
    return offspring

# Single-point crossover function to combine the genes of two parents
def crossover(parent1, parent2, N):
    crosspoint = random.randint(1, N-1)  # Select a random point for crossover
    child1 = copy.deepcopy(parent1)  # Create copies of the parents
    child2 = copy.deepcopy(parent2)
    # Swap the genes after the crossover point between the two parents
    child1.gene[crosspoint:], child2.gene[crosspoint:] = parent2.gene[crosspoint:], parent1.gene[crosspoint:]
    return child1, child2

# Mutation function to slightly modify the genes of an individual
def mutate(individual, mutation_rate, MUTSTEP, MIN, MAX):
    for i in range(len(individual.gene)):
        if random.random() < mutation_rate:  # With a probability of mutation_rate, mutate the gene
            alter = random.uniform(-MUTSTEP, MUTSTEP)  # Generate a small random change (mutation)
            # Apply mutation and ensure the new gene value stays within the MIN and MAX bounds
            individual.gene[i] = max(MIN, min(MAX, individual.gene[i] + alter))

# Main genetic algorithm function
def genetic_algorithm(P, N, T, generations, crossover_rate, mutation_rate, MUTSTEP, MIN, MAX):
    # Initialize population
    population = initialize_population(P, N, MIN, MAX)
    avg_fitness_history = []  # List to track average fitness over generations
    best_fitness_history = []  # List to track best fitness over generations

    for gen in range(generations):
        print(f"Generation {gen + 1}")

        # Select offspring using tournament selection
        offspring_population = tournament_selection(population, T)

        # Apply crossover to create new offspring
        for i in range(0, P, 2):
            if random.random() < crossover_rate and i + 1 < P:  # Perform crossover with probability crossover_rate
                parent1, parent2 = offspring_population[i], offspring_population[i + 1]
                child1, child2 = crossover(parent1, parent2, N)
                offspring_population[i] = child1
                offspring_population[i + 1] = child2

        # Apply mutation to the offspring population
        for ind in offspring_population:
            mutate(ind, mutation_rate, MUTSTEP, MIN, MAX)  # Mutate genes with a small step size
            ind.fitness = evaluate_fitness(ind)  # Recalculate fitness after mutation

        # Replace old population with the new offspring
        population = offspring_population

        # Calculate and store fitness statistics
        avg_fitness = sum(ind.fitness for ind in population) / P  # Average fitness of the population
        best_fitness = min(ind.fitness for ind in population)  # Best fitness in the population
        avg_fitness_history.append(avg_fitness)  # Append average fitness to the history
        best_fitness_history.append(best_fitness)  # Append best fitness to the history

        print(f"Average fitness: {avg_fitness:.2f}, Best fitness: {best_fitness}")

    return avg_fitness_history, best_fitness_history

P = 600  # Smaller population size
N = 20  # Number of genes in each individual (d = 20)
T = 20  # Tournament size for selection
generations = 50  # Number of generations to run the algorithm
crossover_rate = 0.8  # High crossover probability
mutation_rate = 0.1  # Increased mutation rate for better exploration
MUTSTEP = 1.5
  # Moderate mutation step size for gradual changes
MIN = -10.0  # Minimum possible gene value
MAX = 10.0  # Maximum possible gene valu

# Run the genetic algorithm for fitness function 1
avg_fitness_history, best_fitness_history = genetic_algorithm(
    P, N, T, generations, crossover_rate, mutation_rate, MUTSTEP, MIN, MAX
)

# Plot the fitness over generations
plt.figure(figsize=(10, 6))
plt.plot(range(1, generations + 1), avg_fitness_history, label='Average Fitness', marker='o')
plt.plot(range(1, generations + 1), best_fitness_history, label='Best Fitness', marker='x')
plt.title('Fitness Over Generations for Function 1')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.xticks(range(1, generations + 1, 5))  # Set x-ticks for every 5 generations
plt.grid()
plt.legend()
plt.show()
