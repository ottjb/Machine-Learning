import pygame as pg
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

# Define colors
white = (255, 255, 255)
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
        
    def draw(self, surface):
        for p in self.points:
            r = pg.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(surface, self.color, r)
        

def main():
    clock = pg.time.Clock()
    snake = Snake()
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

        screen.fill(black)
        snake.draw(screen)
        # food.draw(screen)
        pg.display.update()
        clock.tick(FRAMES_PER_SECOND)
        
if __name__ == "__main__":
    main()
