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

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLOCK_SIZE = 20
FRAMES_PER_SECOND = 10
col = SCREEN_WIDTH / BLOCK_SIZE
row = SCREEN_HEIGHT / BLOCK_SIZE

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
        if self.getHeadPosition()[0] >= SCREEN_WIDTH or self.getHeadPosition()[0] < 0 or self.getHeadPosition()[1] >= SCREEN_HEIGHT or self.getHeadPosition()[1] < 0:
            self.alive = False
        if self.getHeadPosition() in self.points[1:]:
            self.alive = False
        
    def draw(self, surface):
        for p in self.points:
            r = pg.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(surface, self.color, r)
    
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
        r = pg.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pg.draw.rect(surface, self.color, r)

def main():
    clock = pg.time.Clock()
    snake = Snake()
    food = Food(snake.points)
    # Game Loop
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    snake.turn(0, -1)
                if event.key == K_DOWN:
                    snake.turn(0, 1)
                if event.key == K_LEFT:
                    snake.turn(-1, 0)
                if event.key == K_RIGHT:
                    snake.turn(1, 0)
        snake.move()
        if snake.getHeadPosition() == food.position:
            snake.length += 1
            snake.points.append((0, 0))
            food = Food(snake.points)
        if not snake.alive:
            running = False

        screen.fill(black)
        snake.draw(screen)
        food.draw(screen)
        pg.display.update()
        clock.tick(FRAMES_PER_SECOND)
        
if __name__ == "__main__":
    main()
