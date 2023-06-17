import pygame
import numpy as np
import random
import tensorflow as tf
import Neural_Network
import os
import bisect
import math

# dimensions of the window
WIDTH, HEIGHT = 600, 600
# number of blue points
N_POINTS = 16
# radius of points
RADIUS = 5
# color of network controlled point
NN_POINT_COLOR = (255,0,0)
# color of elastic collision / random point
COLLISION_POINT_COLOR = (0,0,255)
# max speed of points in simulation
MAX_SPEED = 5
# number of neural networks in a population
POPULATION_SIZE = 30

# How many points the NN_Point can 'see' at a time
# based on how close those points are
VISUAL_FIELD = 4
BUILD_SHAPE = (5,5)
WEIGHTS_PATH = "C:\\Users\\ckell\\General\\Programming_Repository\\Partical_Dodge_AI\\Weights"

# Normalization functions
# of the form:  (x - min) / (max - min)
def norm_position(input_position):
    normed = (input_position - 0.0) / (WIDTH - 0.0)
    #print(f"normed position: {normed}")
    return normed

def norm_distance(input_distance):
    max_distance = math.sqrt(2*WIDTH**2)
    normed = (input_distance - 0.0) / (max_distance - 0.0)
    #print(f"normed distance: {normed}")
    return normed

def norm_distance_from_center(input_distance):
    max_distance = math.sqrt(2*(WIDTH/2)**2)
    normed = (input_distance - 0.0) / (max_distance - 0.0)
    #print(f"normed center distance: {normed}")
    return normed

def norm_velocity(input_velocity):
    normed = (abs(input_velocity) - 0.0) / (MAX_SPEED - 0.0)
    #print(f"normed velocity: {normed}")
    return normed

    
# This object spawns elastic particles with random starting position
class Collision_Point:
    def __init__(self):
        initialize_start = random.uniform(0,1)

        #top right corner
        if initialize_start <= 0.25:
            self.x = random.uniform(WIDTH-2*RADIUS, WIDTH-RADIUS)
            self.y = random.uniform(RADIUS, 2*RADIUS)

        #bottom right corner
        if initialize_start > 0.25 and initialize_start <= 0.5:
            self.x = random.uniform(WIDTH-2*RADIUS, WIDTH-RADIUS)
            self.y = random.uniform(HEIGHT-2*RADIUS, HEIGHT-RADIUS)

        #bottom left corner
        if initialize_start > 0.5 and initialize_start <= 0.75:
            self.x = random.uniform(RADIUS, 2*RADIUS)
            self.y = random.uniform(HEIGHT-2*RADIUS, HEIGHT-RADIUS)
        
        #top left corner
        if initialize_start > 0.75 and initialize_start <= 1:
            self.x = random.uniform(RADIUS, 2*RADIUS)
            self.y = random.uniform(RADIUS, 2*RADIUS)

        self.vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.color = COLLISION_POINT_COLOR

    def update(self):
        # update position
        self.x += self.vx
        self.y += self.vy

        # check for collision with walls and reflect if necessary
        if self.x - RADIUS < 0:
            self.vx = -self.vx

        if self.x + RADIUS > WIDTH:
            self.vx = -self.vx

        if self.y - RADIUS < 0:
            self.vy = -self.vy

        if self.y + RADIUS > HEIGHT:
            self.vy = -self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)

# This object is the point that derives its movement information from a neural network.
class NN_Point:
    def __init__(self):
        self.x = WIDTH/2 #random.uniform(RADIUS, WIDTH-RADIUS)
        self.y = HEIGHT/2 #random.uniform(RADIUS, HEIGHT-RADIUS)
        self.vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.color = NN_POINT_COLOR
        self.fitness = 0
        self.brain = Neural_Network.NeuralNetwork()
        self.brain.build((BUILD_SHAPE))
        self.load_best_weights()
        self.alive = True
        self.vision = []

    def load_best_weights(self):
        weights = os.listdir(WEIGHTS_PATH)
        if len(weights) == 0:
            self.previous_best_weights = None
        else:
            weights = sorted(weights, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            self.brain.load_weights(os.path.join(WEIGHTS_PATH,weights[len(weights)-1]))

    def update(self):

        move = self.brain(self.vision)

        # clear vision
        self.vision = []

        # set velocity vectors
        vx = MAX_SPEED*move[0][0]
        vy = MAX_SPEED*move[0][1]

        #print(f"vx: {vx}, vy: {vy}")
        
        # update position
        self.x += vx
        self.y += vy

        if self.x -RADIUS < 0 or self.x + RADIUS > WIDTH:
            self.alive = False
        
        if self.y - RADIUS < 0 or self.y + RADIUS > HEIGHT:
            self.alive = False

    def check_position(self):
        distance_from_center = np.hypot(self.x - WIDTH/2, self.y - HEIGHT/2)
        distance_to_top = HEIGHT - self.y
        distance_to_right = WIDTH - self.x
        distance_to_left = self.x
        distance_to_bottom = self.y
        # [self.x,self.y,distance_to_boundary,distance_from_center]
        return [norm_distance_from_center(distance_from_center),
                norm_distance(distance_to_top),
                norm_distance(distance_to_right),
                norm_distance(distance_to_left),
                norm_distance(distance_to_bottom)]

    # reduce total vision matrix to points closest in visual field
    # output will be tensor supplied to move execution
    def look(self):
        self.vision = self.vision[:VISUAL_FIELD]
        position = self.check_position()
        self.vision.append(position)
        self.vision = np.array(self.vision)
        self.vision = tf.convert_to_tensor(self.vision)

    def reset_position(self):
        self.x = WIDTH/2
        self.y = HEIGHT/2

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)  

def main():

    tf.config.run_functions_eagerly(True)

    while True:
        
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock() 
        # create collision points
        collision_points = [Collision_Point() for _ in range(N_POINTS)]
        # create neural network point
        nn_point = NN_Point()
        # game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((0, 0, 0))
            # Calculate elapsed time
            #elapsed_time = pygame.time.get_ticks() - start_time
            #seconds = elapsed_time // 1000
            nn_point.fitness += 1
            vision = []
            # update and draw blue points
            for point in collision_points:
                #time.sleep(0.01)
                distance = np.hypot(point.x - nn_point.x, point.y - nn_point.y)
                bisect.insort(vision,[distance,point], key = lambda x: x[0])
                point.update()
                point.draw(screen)
                # Creating vision matrix list that contains distance, position, and velocity information
                nn_point.vision.append([norm_distance(distance),
                                        norm_position(point.x),
                                        norm_position(point.y),
                                        norm_velocity(point.vx),
                                        norm_velocity(point.vy)])
                if distance < 2 * RADIUS:
                    #print(f"Blue point collided with red point at ({point.x}, {point.y})")
                    nn_point.alive = False
            # sort the vision matrix to closest first
            nn_point.vision = sorted(nn_point.vision, key = lambda x: x[0])
            for i in range(len(vision)):
                if i < 4:
                    vision[i][1].color = (255,255,0)
                elif i >= 4:
                    vision[i][1].color = (0,0,255)
            # check if point has gone out of bounds or collided with 
            # any other points and terminate iteration if true
            if nn_point.alive == False:
                #print('dead point')
                running = False
            nn_point.look()
            nn_point.update()
            nn_point.draw(screen)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    main()