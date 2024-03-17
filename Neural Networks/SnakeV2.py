import random
import pygame as pg
import numpy as np
from pygame.locals import *
import matplotlib.pyplot as plt

# Game Settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
BLOCK_SIZE = 20
FRAMES_PER_SECOND = 1000
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Neural Network Settings
INPUT_SIZE = (SCREEN_WIDTH // BLOCK_SIZE) * (SCREEN_HEIGHT // BLOCK_SIZE)
HIDDEN1_SIZE = 8
HIDDEN2_SIZE = 8
OUTPUT_SIZE = 4 # Up, Down, Left, Right

# Genetic Algorithm Settings
POPULATION_SIZE = 1000
PARENTS_TO_NEXT_GEN = 100
MUTATION_RATE = 0.05

# Colors
red = (255, 0, 0)
black = (0, 0, 0)
green = (0, 255, 0)

class SnakeGame:
    def __init__(self):
        self.snake = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.lastDirection = None
        self.food = self.generateFood()
        self.timeAlive = 0
        self.foodEaten = 0
        self.collidedWithWall = False
        self.collidedWithSelf = False
        self.stepsWithoutEating = 0
        self.alive = True
        self.turns = 0
    
    def generateFood(self):
        x = random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return (x, y)
    
    def update(self):
        self.timeAlive += 1
        self.stepsWithoutEating += 1
        self.turns += 1
        if self.stepsWithoutEating > 100:
            self.timeAlive *= 0.1
            self.alive = False
            
        if self.direction == UP and self.lastDirection == DOWN:
            self.direction == DOWN
        elif self.direction == DOWN and self.lastDirection == UP:
            self.direction == UP
        elif self.direction == LEFT and self.lastDirection == RIGHT:
            self.direction == RIGHT
        elif self.direction == RIGHT and self.lastDirection == LEFT:
            self.direction == LEFT
        
        if self.direction == UP:
            self.snake.insert(0, (self.snake[0][0], self.snake[0][1] - BLOCK_SIZE))
        elif self.direction == DOWN:
            self.snake.insert(0, (self.snake[0][0], self.snake[0][1] + BLOCK_SIZE))
        elif self.direction == LEFT:
            self.snake.insert(0, (self.snake[0][0] - BLOCK_SIZE, self.snake[0][1]))
        elif self.direction == RIGHT:
            self.snake.insert(0, (self.snake[0][0] + BLOCK_SIZE, self.snake[0][1]))
            
        self.lastDirection = self.direction
            
        if self.turns < 3:
            self.snake.append(((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2)))


        if self.snake[0] == self.food:
            self.foodEaten += 1
            self.stepsWithoutEating = 0
            self.food = self.generateFood()
        else:
            self.snake.pop()
            
        if self.snake[0][0] < 0 or self.snake[0][0] >= SCREEN_WIDTH or self.snake[0][1] < 0 or self.snake[0][1] >= SCREEN_HEIGHT:
            self.collidedWithWall = True
            self.alive = False
            
        if self.snake[0] in self.snake[1:]:
            self.collidedWithSelf = True
            self.alive = False
            
    def draw(self):
        for pos in self.snake:
            r = pg.Rect(pos[0] + 1, pos[1] + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
            pg.draw.rect(screen, green, r)
        pg.draw.rect(screen, red, (self.food[0] + 1, self.food[1] + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        pg.display.update()
        pg.display.set_caption("Time Alive: " + str(self.timeAlive))
        
    def getInputs(self):
        inputs = []
        
        # inputs.extend([self.food[0] - self.snake[0][0],
        #                self.food[1] - self.snake[0][1]])
        
        # inputs.append(self.direction)
        
        # for d in [UP, DOWN, LEFT, RIGHT]:
        #     nextPos = None
        #     if d == UP:
        #         nextPos = (self.snake[0][0], self.snake[0][1] - BLOCK_SIZE)
        #     elif d == DOWN:
        #         nextPos = (self.snake[0][0], self.snake[0][1] + BLOCK_SIZE)
        #     elif d == LEFT:
        #         nextPos = (self.snake[0][0] - BLOCK_SIZE, self.snake[0][1])
        #     elif d == RIGHT:
        #         nextPos = (self.snake[0][0] + BLOCK_SIZE, self.snake[0][1])
                
        #     if nextPos in self.snake or nextPos[0] < 0 or nextPos[0] >= SCREEN_WIDTH or nextPos[1] < 0 or nextPos[1] >= SCREEN_HEIGHT:
        #         inputs.append(1)
        #     else:
        #         inputs.append(0)
                
        # inputs.extend([self.snake[0][1],
        #                SCREEN_HEIGHT - self.snake[0][1],
        #                self.snake[0][0],
        #                SCREEN_WIDTH - self.snake[0][0]])
        
        for i in range(int(SCREEN_WIDTH / BLOCK_SIZE)):
            for j in range(int(SCREEN_HEIGHT / BLOCK_SIZE)):
                if (i * BLOCK_SIZE, j * BLOCK_SIZE) == self.snake[0]:
                    inputs.append(2)
                elif (i * BLOCK_SIZE, j * BLOCK_SIZE) in self.snake[1:]:
                    inputs.append(1)
                elif (i * BLOCK_SIZE, j * BLOCK_SIZE) == self.food:
                    inputs.append(3)
                else: 
                    inputs.append(0)
        return inputs
    
class NeuralNetwork:
    def __init__(self):
        self.input_size = INPUT_SIZE
        self.hidden1_size = HIDDEN1_SIZE
        self.hidden2_size = HIDDEN2_SIZE
        self.output_size = OUTPUT_SIZE
    
        self.weights_input_hidden1 = np.random.rand(self.input_size, self.hidden1_size)
        self.biases_hidden1 = np.random.rand(self.hidden1_size)
        
        self.weights_hidden1_hidden2 = np.random.rand(self.hidden1_size, self.hidden2_size)
        self.biases_hidden2 = np.random.rand(self.hidden2_size)
        
        self.weights_hidden2_output = np.random.rand(self.hidden2_size, self.output_size)
        self.output_biases = np.random.rand(self.output_size)
    
    def predict(self, inputs):
        hidden1 = np.dot(inputs, self.weights_input_hidden1) + self.biases_hidden1
        hidden1 = self.tanh(hidden1)
        
        hidden2 = np.dot(hidden1, self.weights_hidden1_hidden2) + self.biases_hidden2
        hidden2 = self.sigmoid(hidden2)
        
        output = np.dot(hidden2, self.weights_hidden2_output) + self.output_biases
        output = self.softmax(output)
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=0)
    
def crossover(parent1, parent2) -> NeuralNetwork:
    child1 = NeuralNetwork()
    child2 = NeuralNetwork()
    
    if random.random() < 0.5:
        parent1, parent2 = parent2, parent1
    # Crossover weights_input_hidden1
    crossoverPoint = random.randint(0, parent1.input_size * parent1.hidden1_size)
    
    child1.weights_input_hidden1 = np.concatenate((parent1.weights_input_hidden1[:crossoverPoint], parent2.weights_input_hidden1[crossoverPoint:]), axis=0)
    child2.weights_input_hidden1 = np.concatenate((parent2.weights_input_hidden1[:crossoverPoint], parent1.weights_input_hidden1[crossoverPoint:]), axis=0)
    
    # Crossover hidden1_biases
    crossoverPoint = random.randint(0, parent1.hidden1_size)
    child1.biases_hidden1 = np.concatenate((parent1.biases_hidden1[:crossoverPoint], parent2.biases_hidden1[crossoverPoint:]), axis=0)
    child2.biases_hidden1 = np.concatenate((parent2.biases_hidden1[:crossoverPoint], parent1.biases_hidden1[crossoverPoint:]), axis=0)
    
    # Crossover weights_hidden1_hidden2
    crossoverPoint = random.randint(0, parent1.hidden1_size * parent1.hidden2_size)
    
    child1.weights_hidden1_hidden2 = np.concatenate((parent1.weights_hidden1_hidden2[:crossoverPoint], parent2.weights_hidden1_hidden2[crossoverPoint:]), axis=0)
    child2.weights_hidden1_hidden2 = np.concatenate((parent2.weights_hidden1_hidden2[:crossoverPoint], parent1.weights_hidden1_hidden2[crossoverPoint:]), axis=0)
    
    # Crossover hidden2_biases
    crossoverPoint = random.randint(0, parent1.hidden2_size)
    child1.biases_hidden2 = np.concatenate((parent1.biases_hidden2[:crossoverPoint], parent2.biases_hidden2[crossoverPoint:]), axis=0)
    child2.biases_hidden2 = np.concatenate((parent2.biases_hidden2[:crossoverPoint], parent1.biases_hidden2[crossoverPoint:]), axis=0)
    
    # Crossover weights_hidden2_output
    crossoverPoint = random.randint(0, parent1.hidden2_size * parent1.output_size)
    child1.weights_hidden2_output = np.concatenate((parent1.weights_hidden2_output[:crossoverPoint], parent2.weights_hidden2_output[crossoverPoint:]), axis=0)
    child2.weights_hidden2_output = np.concatenate((parent2.weights_hidden2_output[:crossoverPoint], parent1.weights_hidden2_output[crossoverPoint:]), axis=0)
    
    # Crossover output_biases
    crossoverPoint = random.randint(0, parent1.output_size)
    child1.output_biases = np.concatenate((parent1.output_biases[:crossoverPoint], parent2.output_biases[crossoverPoint:]), axis=0)
    child2.output_biases = np.concatenate((parent2.output_biases[:crossoverPoint], parent1.output_biases[crossoverPoint:]), axis=0)
    
    return child1, child2

    
def mutate(nn):
    network = nn
    # Mutate weights_input_hidden1
    for i in range(network.input_size):
        for j in range(network.hidden1_size):
            if random.random() < MUTATION_RATE:
                network.weights_input_hidden1[i][j] = np.random.normal(-1, 1)
                
    # Mutate biases_hidden1
    for i in range(network.hidden1_size):
        if random.random() < MUTATION_RATE:
            network.biases_hidden1[i] = np.random.normal(-1, 1)
            
    # Mutate weights_hidden1_hidden2
    for i in range(network.hidden1_size):
        for j in range(network.hidden2_size):
            if random.random() < MUTATION_RATE:
                network.weights_hidden1_hidden2[i][j] = np.random.normal(-1, 1)
                
    # Mutate biases_hidden2
    for i in range(network.hidden2_size):
        if random.random() < MUTATION_RATE:
            network.biases_hidden2[i] = np.random.normal(-1, 1)
            
    # Mutate weights_hidden2_output
    for i in range(network.hidden2_size):
        for j in range(network.output_size):
            if random.random() < MUTATION_RATE:
                network.weights_hidden2_output[i][j] = np.random.normal(-1, 1)
    
    # Mutate output_biases
    for i in range(network.output_size):
        if random.random() < MUTATION_RATE:
            network.output_biases[i] = np.random.normal(-1, 1)
    return network
    

def fitness(network):
    game = SnakeGame()
    while game.alive:
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                exit()
        inputs = game.getInputs()
        out = network.predict(inputs)
        direction = np.argmax(out)
        game.direction = direction
        game.update()
        
        screen.fill(black)
        game.draw()
        pg.display.flip()
        clock.tick(FRAMES_PER_SECOND)
    #fitness = game.foodEaten * 15 + game.timeAlive * 2
    #if game.collidedWithSelf: fitness - 50
    #elif game.collidedWithWall: fitness - 100
    return game.timeAlive + 2**game.foodEaten


pg.init()
screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pg.time.Clock()

population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
gen = 1
while True:
    print("Generation:", gen)
    fitness_scores = [fitness(nn) for nn in population]
    print("Maximum fitness of generation " + str(gen) + ": " + str(np.max(fitness_scores)))
    print("Mean fitness of generation " + str(gen) + ": " + str(np.mean(fitness_scores)))
    
    parents = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda x: fitness_scores[x], reverse=True)[:PARENTS_TO_NEXT_GEN]]
    
    offspring = []
    while len(offspring) < POPULATION_SIZE - len(parents):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        if parent1 != parent2:
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

    offspring = [mutate(nn) for nn in offspring]
    population = parents
    for n in offspring:
        population.append(n)
    gen += 1
