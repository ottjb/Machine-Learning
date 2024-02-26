import random as r

# Declaring variables for the knapsack problem
capacity = 10
numOfItems = 5
minWeight = 1
maxWeight = 6
minValue = 1
maxValue = 10

# Declaring variables for the genetic algorithm
populationSize = 10
population = []
crossoverRate = 0.9
mutationRate = 0.1
generations = 100

# Create the items that can be put in the knapsack
weight = [0] * numOfItems
value = [0] * numOfItems
for i in range(numOfItems): 
    weight[i] = r.randint(minWeight, maxWeight)
    value[i] = r.randint(minValue, maxValue)
    
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
    
print("The initial population is: ", population)

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
    
population.sort(key = lambda x: fitness(x))

print("The sorted population is: ", population)

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

def tournamentSelection(candidates):
    while len(candidates) > 1:
        for i in range(len(candidates) / 2):
            print("here" + str(i))
            if candidates[i] < candidates[i + 1]:
                candidates.pop(i)
            else:
                candidates.pop(i + 1)
    return candidates[0]
                
candidates = population[:10]
print(candidates)
print(tournamentSelection(candidates))
