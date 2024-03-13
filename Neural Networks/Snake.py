import numpy as np
import pygame as pg
import random as r
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    KEYDOWN,
    QUIT
)

SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
BLOCK_SIZE = 20
col = SCREEN_WIDTH // BLOCK_SIZE
row = SCREEN_HEIGHT // BLOCK_SIZE
FRAMES_PER_SECOND = 10

snake = None
food = None

INPUT_SIZE = 7
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 4
POPULATION_SIZE = 5
MUTATION_RATE = 0.01

# Define colors
red = (255, 0, 0)
black = (0, 0, 0)
green = (0, 255, 0)

pg.init()
pg.display.set_caption("Snake Game")
screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Snake():
    def __init__(self, color=green):
        self.length = 1
        self.points = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = (1, 0)
        self.color = color
        self.alive = True
        self.weights_ih = np.random.rand(INPUT_SIZE, HIDDEN1_SIZE)
        self.weights_hh = np.random.rand(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.weights_ho = np.random.rand(HIDDEN2_SIZE, OUTPUT_SIZE)
        
    def getHeadPosition(self):
        return self.points[0]
    
    def getDirection(self):
        return self.direction
        
    def turn(self, x, y):
        self.direction = (x, y)
        
    def move(self):
        currentX, currentY = self.getHeadPosition()
        directionX, directionY = self.direction
        newHeadPosition = ((currentX + directionX * BLOCK_SIZE), (currentY + directionY * BLOCK_SIZE))
        self.points.insert(0, newHeadPosition)
        self.points.pop()
        if self.getHeadPosition()[0] < 0 or self.getHeadPosition()[0] >= SCREEN_WIDTH or self.getHeadPosition()[1] < 0 or self.getHeadPosition()[1] >= SCREEN_HEIGHT or self.getHeadPosition() in self.points[1:]:
            self.alive = False

    def draw(self, surface):
        for p in self.points:
            r = pg.Rect((p[0] + 1, p[1] + 1), (BLOCK_SIZE - 1, BLOCK_SIZE - 1))
            pg.draw.rect(surface, self.color, r)
            
    def feedforward(self, inputs):
        self.hidden1 = np.dot(inputs, self.weights_ih)
        self.hidden1_activation = self.sigmoid(self.hidden1)
        
        self.hidden2 = np.dot(self.hidden1_activation, self.weights_hh)
        self.hidden2_activation = self.sigmoid(self.hidden2)
        
        self.output = np.dot(self.hidden2_activation, self.weights_ho)
        self.output_activation = self.softmax(self.output)
        return self.output_activation
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, keepdims=True)
        return probabilities
    
    # for i in range to check a certain distance in each direction
    def gatherInputs(self, snakeHeadPosition, foodPosition):
        '''
        foodInputs = []
        wallInputs = []
        bodyInputs = []
        directionInputs = []
        vision = 50
        wallVision = 3
        foodInVision = False
        wallInVision = False
        bodyInVision = False
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            for i in range(1, vision + 1):
                x = snakeHeadPosition[0] + dx * i * BLOCK_SIZE
                y = snakeHeadPosition[1] + dy * i * BLOCK_SIZE
                if (x, y) == foodPosition:
                    foodInVision = True
                if (x, y) in self.points:
                    bodyInVision = True

            for i in range(1, wallVision + 1):
                x = snakeHeadPosition[0] + dx * i * BLOCK_SIZE
                y = snakeHeadPosition[1] + dy * i * BLOCK_SIZE
                if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
                    wallInVision = True

            if foodInVision:
                foodInputs.append(1)
                foodInVision = False
            else:
                foodInputs.append(0)

            if wallInVision:
                wallInputs.append(1)
                wallInVision = False
            else:
                wallInputs.append(0)
            
            if bodyInVision:
                bodyInputs.append(1)
                bodyInVision = False
            else:
                bodyInputs.append(0)

        for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
            if (dx, dy) == self.direction:
                directionInputs.append(1)
            else:
                directionInputs.append(0)
            
        return foodInputs + wallInputs + bodyInputs + directionInputs
        '''
        

        distanceFromWalls = [snakeHeadPosition[0] // BLOCK_SIZE, SCREEN_WIDTH - snakeHeadPosition[0], snakeHeadPosition[1], SCREEN_HEIGHT - snakeHeadPosition[1]]

        foodVector_x = foodPosition[0] - snakeHeadPosition[0]
        foodVector_y = foodPosition[1] - snakeHeadPosition[1]
        directionOfFood = np.degrees(np.arctan2(foodVector_y, foodVector_x))
        if directionOfFood < 0:
            directionOfFood += 360

        # Get 2d array of the snake's body
        # body = np.zeros((int(SCREEN_WIDTH / BLOCK_SIZE), int(SCREEN_HEIGHT / BLOCK_SIZE)))
        # for i in range(col):
        #     for j in range(row):
        #         x = i * BLOCK_SIZE
        #         y = j * BLOCK_SIZE
        #         if (x, y) == self.getHeadPosition():
        #             body[i][j] = 2
        #         elif (x, y) in self.points:
        #             body[i][j] = 1
        #         elif (x, y) == foodPosition:
        #             body[i][j] = 3
        #         else:
        #             body[i][j] = 0
        
        # Direction of the snake
        direction = None
        if self.direction == (-1, 0):
            direction = 0
        elif self.direction == (1, 0):
            direction = 1
        elif self.direction == (0, -1):
            direction = 2
        elif self.direction == (0, 1):
            direction = 3

        proximityToTail = 0
        for point in self.points:
            distance = abs(snakeHeadPosition[0] - point[0]) + abs(snakeHeadPosition[1] - point[1])
            proximityToTail = min(proximityToTail, distance)

        input = distanceFromWalls + [directionOfFood]
        # for i in range(col):
        #     for j in range(row):
        #         input.append(body[i][j])
        input.append(direction)
        input.append(proximityToTail)
        return input
        

        
    
    
class Food():
    def __init__(self, snakePoints, color=red):
        self.position = self.findNewPosition(snakePoints)
        self.color = color
    
    def findNewPosition(self, snakePoints) -> tuple:
        x = r.randint(0, col - 1) * BLOCK_SIZE
        y = r.randint(0, row - 1) * BLOCK_SIZE
        while (x, y) in snakePoints:
            x = r.randint(0, col - 1) * BLOCK_SIZE
            y = r.randint(0, row - 1) * BLOCK_SIZE
        return (x, y)

    def draw(self, surface):
        r = pg.Rect((self.position[0] + 1, self.position[1] + 1), (BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        pg.draw.rect(surface, self.color, r)
        

def runGame(snake : Snake):
    clock = pg.time.Clock()
    snake = snake
    food = Food(snake.points)
    lastDirection = (0, 0)
    direction = snake.getDirection()
    framesAlive = 0
    # Game Loop
    running = True
    while running:
        inputs = snake.gatherInputs(snake.getHeadPosition(), food.position)
        outputs = snake.feedforward(inputs)
        print(inputs)
        print(outputs)
        max_output_index = np.argmax(outputs)
        # Convert the index to a direction
        if max_output_index == 0:
            direction = (-1, 0)
        elif max_output_index == 1:
            direction = (1, 0)
        elif max_output_index == 2:
            direction = (0, -1)
        elif max_output_index == 3:
            direction = (0, 1)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_UP and lastDirection != (0, 1):
                    direction = (0, -1)
                if event.key == K_DOWN and lastDirection != (0, -1):
                    direction = (0, 1)
                if event.key == K_LEFT and lastDirection != (1, 0):
                    direction = (-1, 0)
                if event.key == K_RIGHT and lastDirection != (-1, 0):
                    direction = (1, 0)
        snake.turn(direction[0], direction[1])
        snake.move()
        lastDirection = direction
        if snake.getHeadPosition() == food.position:
            snake.length += 1
            snake.points.append((food.position[0], food.position[1]))
            food = Food(snake.points)
        if not snake.alive:
            return (framesAlive, snake.length - 1)

        screen.fill(black)
        snake.draw(screen)
        food.draw(screen)
        pg.display.update()
        framesAlive += 1
        clock.tick(FRAMES_PER_SECOND)
        
def main():
    for i in range(POPULATION_SIZE):
        snake = Snake()
        runGame(snake)

        
if __name__ == "__main__":
    main()
