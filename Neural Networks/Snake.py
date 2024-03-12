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

INPUT_SIZE = 28
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 4
POPULATION_SIZE = 100
MUTATION_RATE = 0.01

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLOCK_SIZE = 20
FRAMES_PER_SECOND = 10
col = SCREEN_WIDTH / BLOCK_SIZE
row = SCREEN_HEIGHT / BLOCK_SIZE

snake = None
food = None

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
        self.direction = (0, 0)
        self.color = color
        self.alive = True
        self.weights_ih = np.random.rand(INPUT_SIZE, HIDDEN1_SIZE)
        self.weights_hh = np.random.rand(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.weights_ho = np.random.rand(HIDDEN2_SIZE, OUTPUT_SIZE)
        
    def getHeadPosition(self):
        return self.points[0]
        
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
        self.output_activation = self.sigmoid(self.output)
        return self.output_activation
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # for i in range to check a certain distance in each direction
    def gatherInputs(self, snakeHeadPosition, foodPosition):
        inputs = []
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            inputs.append(int((foodPosition[0] + dx * BLOCK_SIZE, foodPosition[1] + dy * BLOCK_SIZE) == snakeHeadPosition))
        return inputs
    
    
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
    print(snake.getHeadPosition(), food.position)
    lastDirection = (0, 0)
    direction = (0, 0)
    # Game Loop
    running = True
    while running:
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
            running = False
            
        inputs = snake.gatherInputs(snake.getHeadPosition(), food.position)
        print(inputs)

        screen.fill(black)
        snake.draw(screen)
        food.draw(screen)
        pg.display.update()
        clock.tick(FRAMES_PER_SECOND)
        
def main():
    snake = Snake()
    
    runGame(snake)
        
if __name__ == "__main__":
    main()
