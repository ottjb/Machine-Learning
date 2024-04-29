import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

# Directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Font Settings
FONT = pygame.font.Font("freesansbold.ttf", 35)

# Game Constants
BLOCK_SIZE = 20
SPEED = 50

class Snake:
    def __init__(self, w=320, h=320):
        self.w = w
        self.h = h
        
        self.games = 0
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self._reset()
        
    def _reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frames = 0
        self.games += 1
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frames += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frames > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            self._place_food()
            reward = 10
        else:
            self.snake.pop()
            
        self._update()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _update(self):
        self.display.fill(BLACK)
        
        for p in self.snake:
            r = pygame.Rect(p.x + 1, p.y + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
            pygame.draw.rect(self.display, GREEN, r)
        pygame.draw.rect(self.display, RED, (self.food.x + 1, self.food.y + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        
        text = FONT.render("Game: " + str(self.games), True, WHITE)
        self.display.blit(text, [0, 0])
        text = FONT.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 50])
        
        pygame.display.flip()
    
    def _move(self, action):
        # [Straight, Right, Left]
        direction_order = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = direction_order.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_direction = direction_order[index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = direction_order[next_index]
        else:
            next_index = index - 1
            new_direction = direction_order[next_index]
        
        self.direction = new_direction
        
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    
    