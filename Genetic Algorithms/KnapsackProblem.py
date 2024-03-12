import random as r
import math as m
import matplotlib.pyplot as plt

# Declaring variables for the knapsack problem
numOfItems = 8
minWeight = 1
maxWeight = 6
minValue = 1
maxValue = 10
capacity = 11

# Declaring variables for the genetic algorithm
populationSize = 20
population = []
crossoverRate = 0.9
mutationRate = 0.1
generations = 10
averageFitnessOfGenerations = []

# Create the items that can be put in the knapsack
weight = [0] * numOfItems
value = [0] * numOfItems
for i in range(numOfItems): 
    weight[i] = r.randint(minWeight, maxWeight)
    value[i] = r.randint(minValue, maxValue)
    
print("The maximum weight of the knapsack is: " + str(capacity))
print("The weight of the items is: ", weight)
print("The value of the items is: ", value)

# Create the initial population
def createCandidate():
    candidate = ''
    for i in range(numOfItems):
        candidate += str(r.randint(0, 1))
    return candidate

for i in range(populationSize):
    population.append(createCandidate())
    
#print("The initial population is: ", population)

# Find the fitness of each candidate and sort the population based on the fitness
'''
This fitness function with find the total weight and value of each candidate.
If the total weight of the candidate is greater than the knapsacks capacity,
the fitness is automatically set to 0. Otherwise, the fitness is set to the 
total value. I plan to change the fitness function to not only just look at 
total value, but also how much of the knapsack's capacity is utilized.
'''
def fitness(candidate):
    totalWeight = 0
    totalValue = 0
    for i in range(len(candidate)):
        if candidate[i] == '1':
            totalWeight += weight[i]
            totalValue += value[i]
    if totalWeight > capacity:
        return 0
    else:
        return totalValue
    
def findAverageFitness():
    totalFitness = 0
    for i in range(populationSize):
        totalFitness += fitness(population[i])
    avg = totalFitness / populationSize
    averageFitnessOfGenerations.append(avg)
    return str(avg)
    
#population.sort(key = lambda x: fitness(x))

#print("The sorted population is: ", population)

# Create the next generation
'''
The crossover function randomly selects a "spliting point," and splits both
candidates at this point. The new candidate will consist of the first
candidate' selections before the splitting point, and the second candidate's
selections after the splitting point.
'''
def crossover(candidate1, candidate2):
    split = r.randint(1, numOfItems - 1)
    newCandidate = candidate1[:split] + candidate2[split:]
    return newCandidate

'''
The mutate function randomly selects one of the candidate's indexes and swaps
it to 0 if it is 1 and to 1 if it is 0.
'''
def mutate(candidate):
    mutatedIndex = r.randint(0, numOfItems - 1)
    if candidate[mutatedIndex] == '1':
        newCandidate = candidate[:mutatedIndex] + '0' + candidate[mutatedIndex + 1:]
    else:
        newCandidate = candidate[:mutatedIndex] + '1' + candidate[mutatedIndex + 1:]
    return newCandidate

'''
The tournamentSelection will take a random sample of the population as a
parameter and return the candidate with the highest fitness. This function
will be ran multiple times to create the next generation.
'''
def tournamentSelection():
    r.shuffle(population)
    candidates = population[:m.floor(populationSize * .1)]
    winner = candidates[0]
    for i in range(1, len(candidates)):
        if fitness(candidates[i]) > fitness(winner):
            winner = candidates[i]
    return winner

'''
The createNewCandidate function will create a new candidate based on the
crossover and mutation rates. If the crossover rate is met, the function will
create a new candidate by crossing over two parents. If the mutation rate is
met, the function will mutate the new candidate. If neither rate is met, the
function will create a new candidate by selecting a random candidate from the
population.
'''
def createNewChild():
    doCrossover = False
    doMutate = False
    if r.random() < crossoverRate:
        crossover = True
    if r.random() < mutationRate:
        mutate = True
    if doCrossover:
        parent1 = tournamentSelection()
        parent2 = tournamentSelection()
        child = crossover(parent1, parent2)
        if doMutate:
            return mutate(child)
        else:
            return child
    else:
        child = tournamentSelection()
        if doMutate:
            return mutate(child)
        else:
            return child
        
# Create the next generation
'''
The createNewGeneration function will create a new generation of candidates
based on the createNewChild function.
'''
def createNewGeneration():
    global population
    newPopulation = []
    for i in range(populationSize):
        newPopulation.append(createNewChild())
    population = newPopulation

# Print the generations stats
def displayGeneration(currentGeneration):
    print("---Generation " + str(currentGeneration) + "---")
    population.sort(key = lambda x : fitness(x), reverse=True)
    print("Top ten candidates: ", population[:10])
    print("The fitness of the best candidate is: ", fitness(population[0]))
    print("The average fitness of this generation is: " + findAverageFitness())

# Main loop
for i in range(generations - 1):
    displayGeneration(i + 1)
    createNewGeneration()
displayGeneration(generations)

# Print the best candidate
population.sort(key = lambda x : fitness(x), reverse=True)
print("The optimal solution is: ", population[0])

# Plot the average fitness of each generation
plt.plot(averageFitnessOfGenerations)
plt.xlabel('Generation')
plt.xticks(range(0, generations))
plt.ylabel('Average Fitness')
plt.yticks(range(0, maxValue * numOfItems, 10))
plt.title('Average Fitness of Each Generation')
plt.show()
print(len(averageFitnessOfGenerations))
