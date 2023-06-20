import pygame
import numpy as np
import random
import tensorflow as tf
import Neural_Network
import os
import bisect
import math
import threading

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
POPULATION_SIZE = 100

# How many points the NN_Point can 'see' at a time
# based on how close those points are
VISUAL_FIELD = 4
BUILD_SHAPE = (5,5)
WEIGHTS_PATH = "C:\\Users\\ckell\\General\\Programming_Repository\\Partical_Dodge_AI\\Weights"

# Normalization functions
# of the form:  (x - min) / (max - min)
def norm_position(input_position):
    #normalized to center of screen space
    normed = (input_position-WIDTH - 0.0) / (WIDTH/2 - 0.0)
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


# Network_Pool Class
# This object will initialize a random population, track the best performers, and spawn
# new networks. 
class Network_Pool:
    def __init__(self):
        self.population = []
        self.rattle_intensity = 0.1
        self.generation_number = 0
        self.previous_best_fitness = 0
        self.current_best_fitness = 0
        self.current_second_best_fitness = 0
        self.previous_best_weights = False

        self.check_evolution_progress()
        self.find_best_previous_weights()
        self.generate_new_population()
        self.load_previous_best_weights()
        
    # generate new population with random point initialization
    # if applicable, weights are loaded in later
    def generate_new_population(self):
        for _ in range(POPULATION_SIZE):
            nn_point = NN_Point()
            nn_point.brain.build(BUILD_SHAPE)
            self.population.append(nn_point)
        
    # check for parameters from previous iterations
    # if none instantiate variables
    def check_evolution_progress(self):
        params_dict = {}
        if os.path.isfile('Parameters.txt'):
            with open('Parameters.txt','r') as params:
                for line in params:
                    line = line.split(',')
                    params_dict[line[0]] = line[1]
            
            self.rattle_intensity = float(params_dict['rattle_intensity'])
            self.generation_number = int(params_dict['generation_number'])
        
        else:
            print('No evolution history present. Starting evolution process.')

    # record this generation's metrics
    def record_evolution_progress(self):
        with open('Parameters.txt','w') as params:
            params.writelines(f"rattle_intensity,{self.rattle_intensity}\n")
            params.writelines(f"generation_number,{self.generation_number}\n")


    # fix for resetting population as bug occurs when carrying initial population 
    # of NN_Point objects over to next population cycle
    def reset_population(self):
        BrainBasket = []
        for i in range(POPULATION_SIZE):
            BrainBasket.append(self.population[i].brain)
        
        self.population = []
        for i in range(POPULATION_SIZE):
            nn_point = NN_Point()
            nn_point.brain = BrainBasket[i]
            self.population.append(nn_point)

    # saves the best performing neural network of the population
    # as its fitness score to be searchable later
    def save_best(self):
        print(f"Saving best weights. Fitness = {self.population[0].fitness}")
        self.population[0].brain.save_weights(f"weights\w_{self.population[0].fitness}.h5")

    # set the location of the best previous weights file
    def find_best_previous_weights(self):
        weights = os.listdir(WEIGHTS_PATH)

        if len(weights) == 0:
            self.previous_best_weights = None

        else:
            weights = sorted(weights, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            self.previous_best_fitness = int(weights[0].split('_')[1].split('.')[0])
            self.previous_best_weights = os.path.join(WEIGHTS_PATH,weights[len(weights)-1])

    # loads the previous best weights recorded into the population
    def load_previous_best_weights(self):
        if self.previous_best_weights != None:
            for i in range(len(self.population)):
                self.population[i].brain.load_weights(self.previous_best_weights)
        else:
            print('No prior weights detected.')

    # orders the NN_Point objects by their fitness attribute
    # useful for selecting parents
    def order_by_performance(self):
        sorted_networks = sorted(self.population, key=lambda network: network.fitness, reverse=True)
        self.population = sorted_networks
        self.current_best_fitness = self.population[0].fitness
        self.current_second_best_fitness = self.population[1].fitness

    # the entirety of the evolution process.
    # checks if the current fitness is better than previous
    # if yes, mutate from current and if no mutate from previous generation
    # then adjust rattle mutation intenstity and crossover parent selection
    def evolve(self):

        # order the networks in the population by their fitness 
        self.order_by_performance()

        # set previous best fitness every generation
        self.find_best_previous_weights()

        # increase generation number
        self.generation_number += 1

        # check if both current and second best fitness are greater than previous gen best
        # decrease rattle intensity to promote convergence then perform standard
        # crossover and mutation with best performers 
        print(f"Current Best: {self.current_best_fitness}, Current Second Best: {self.current_second_best_fitness}, Previous Best: {self.previous_best_fitness}, Rattle Intensity: {self.rattle_intensity}")
        if self.current_best_fitness > self.previous_best_fitness and self.current_second_best_fitness > self.previous_best_fitness:
            print('Evolution Case 1')
            self.save_best()
            #self.rattle_intensity = self.rattle_intensity * 0.9
            self.population[0].brain = self.crossover(self.population[0].brain,self.population[1].brain)
            self.mutate_population(self.rattle(self.population[0].brain,POPULATION_SIZE))

        # check if current best is better than previous but second best is not
        # in this case crossover with best and previous generation best
        if self.current_best_fitness > self.previous_best_fitness and self.current_second_best_fitness < self.previous_best_fitness:
            print('Evolution Case 2')
            self.save_best()
            #self.rattle_intensity = self.rattle_intensity * 0.95
            self.population[1].brain.load_weights(self.previous_best_weights)
            self.population[0].brain = self.crossover(self.population[0].brain,self.population[1].brain)
            self.mutate_population(self.rattle(self.population[0].brain,POPULATION_SIZE))
        
        # check if current best is not better than previous generation
        # in this case slightly increase rattle intensity and mutate
        # from previous generation
        if self.current_best_fitness <= self.previous_best_fitness:
            print('Evolution Case 3')
            #self.rattle_intensity = self.rattle_intensity * 1.001
            self.load_previous_best_weights()
            self.mutate_population(self.rattle(self.population[0].brain,POPULATION_SIZE))

        self.record_evolution_progress()

    def crossover(self, network1, network2):
        # Create a new instance of the network
        child_network = Neural_Network.NeuralNetwork()
        child_network.build(BUILD_SHAPE)

        # Get the weights of the parent networks
        parent1_weights = network1.get_weights()
        parent2_weights = network2.get_weights()

        # Perform crossover by randomly selecting weights from each parent
        child_weights = []
        for i in range(len(parent1_weights)):
            parent1_weight = parent1_weights[i]
            parent2_weight = parent2_weights[i]

            # Randomly choose which parent's weight to select
            mask = np.random.choice([0, 1], size=parent1_weight.shape)
            child_weight = np.where(mask, parent1_weight, parent2_weight)
            child_weights.append(child_weight)

        # Set the weights of the child network
        child_network.set_weights(child_weights)

        return child_network
    
    def rattle(self, parent_network, num_offspring):
        offspring_networks = []
        for _ in range(num_offspring):
            # Create a new instance of the network
            offspring = Neural_Network.NeuralNetwork()
            offspring.build(BUILD_SHAPE)
            offspring.set_weights(parent_network.get_weights())

            # Perform mutation on the offspring network's weights
            weights = offspring.get_weights()
            mutated_weights = [weight + np.random.uniform(-self.rattle_intensity, self.rattle_intensity, size=weight.shape) for weight in weights]

            # Assign the mutated weights back to the offspring network
            offspring.set_weights(mutated_weights)

            offspring_networks.append(offspring)

        return offspring_networks
    
    # takes a list of neural networks and applies them to the population pool
    def mutate_population(self, mutated_brains):
        for i in range(len(mutated_brains)):
            self.population[i].brain = mutated_brains[i]

    def set_rattle_intensity_from_tier(self):
        if self.previous_best_fitness < 500:
            self.rattle_intensity = 0.15
        if self.previous_best_fitness < 1000 and self.previous_best_fitness >= 500:
            self.rattle_intensity = 0.1
        if self.previous_best_fitness < 1500 and self.previous_best_fitness >= 1000:
            self.rattle_intensity = 0.08
        if self.previous_best_fitness < 2000 and self.previous_best_fitness >= 1500:
            self.rattle_intensity = 0.05
        if self.previous_best_fitness < 2500 and self.previous_best_fitness >= 2000:
            self.rattle_intensity = 0.03
        if self.previous_best_fitness < 3000 and self.previous_best_fitness >= 2500:
            self.rattle_intensity = 0.01
    
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
        self.alive = True
        self.vision = []

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
        #return [distance_from_center,distance_to_top,distance_to_right,distance_to_left,distance_to_bottom]
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

def game_function(i,pool):
    # iterate through population and run simulation
    """
    if i == len(pool.population)-1:
        print(f'iteration {i} start')
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock() 
        # create collision points
        collision_points = [Collision_Point() for _ in range(N_POINTS)]
        # create neural network point
        nn_point = pool.population[i]
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
                nn_point.vision.append([distance,point.x,point.y,point.vx,point.vy])
                
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
        """
    if True:
        print(f'iteration {i} start')
        # create collision points
        collision_points = [Collision_Point() for _ in range(N_POINTS)]
        # create neural network point
        nn_point = pool.population[i]
        # game loop
        running = True
        while running:

            # Calculate elapsed time
            #elapsed_time = pygame.time.get_ticks() - start_time
            #seconds = elapsed_time // 1000
            nn_point.fitness += 1

            # update and draw blue points
            for point in collision_points:
                #time.sleep(0.01)
                distance = np.hypot(point.x - nn_point.x, point.y - nn_point.y)
                point.update()

                # Creating vision matrix list that contains distance, position, and velocity information
                #nn_point.vision.append([distance,point.x,point.y,point.vx,point.vy])
                
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

            # check if point has gone out of bounds or collided with 
            # any other points and terminate iteration if true
            if nn_point.alive == False:
                #print('dead point')
                running = False

            nn_point.look()
            nn_point.update()

def main():

    pool = Network_Pool()

    tf.config.run_functions_eagerly(True)

    while True:
        threads = []
        for i in range(len(pool.population)):
            t = threading.Thread(target=game_function,args=(i,pool))
            t.start()
            threads.append(t)
            #game_function(i,pool)

        for t in threads:
            t.join()

        print('pop end')
        pool.evolve()
        pool.reset_population()
        pool.set_rattle_intensity_from_tier()

if __name__ == "__main__":
    main()