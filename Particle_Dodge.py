import pygame
import numpy as np
import random
import tensorflow as tf
import Neural_Network
import os

# dimensions of the window
WIDTH, HEIGHT = 600, 600
# number of blue points
N_POINTS = 10
# radius of points
RADIUS = 10
# color of network controlled point
NN_POINT_COLOR = (255,0,0)
# color of elastic collision / random point
COLLISION_POINT_COLOR = (0,0,255)
# max speed of points in simulation
MAX_SPEED = 10
# number of neural networks in a population
POPULATION_SIZE = 3

# path to current best network weights
BEST_NETWORK = 'model_weights.h5'
# sample neural network input. Will change later.
NN_INPUT = np.random.rand(1,8)

# Network_Pool Class
# This object will initialize a random population, track the best performers, and spawn
# new networks. 
class Network_Pool:
    def __init__(self):
        self.population = []

        for i in range(POPULATION_SIZE):
            self.population.append(NN_Point())

    def save_best(self):
        self.population[0].brain.save_weights('model_weights.h5')

    def load_network(self):
        self.population[0].brain.load_weights('model_weights.h5')

    def order_by_performance(self):
        sorted_networks = sorted(self.population, key=lambda network: network.fitness)
        self.population = sorted_networks

    def crossover(self, network1, network2):
        # Create a new instance of the network
        child_network = Neural_Network.NeuralNetwork()
        child_network.build((None,8))

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
            offspring.build((None,8))
            offspring.set_weights(parent_network.get_weights())

            # Perform mutation on the offspring network's weights
            weights = offspring.get_weights()
            mutated_weights = [weight + np.random.uniform(-0.1, 0.1, size=weight.shape) for weight in weights]

            # Assign the mutated weights back to the offspring network
            offspring.set_weights(mutated_weights)

            offspring_networks.append(offspring)

        return offspring_networks
    
    # takes a list of neural networks and applies them to the population pool
    def mutate_population(self, mutated_brains):
        for i in range(len(mutated_brains)):
            self.population[i].brain = mutated_brains[i]

    def reset_population_attributes(self):
        for i in range(len(self.population)):
            self.population[i].reset_position()
    
# This object spawns mostly elastic particles with slight random behavior.
class Collision_Point:
    def __init__(self):
        self.x = random.uniform(RADIUS, WIDTH-RADIUS)
        self.y = random.uniform(RADIUS, HEIGHT-RADIUS)
        self.vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        self.color = COLLISION_POINT_COLOR

    def update(self):
        # update position
        self.x += self.vx
        self.y += self.vy

        # check for collision with walls and reflect if necessary
        # includes slight random collision effect
        if self.x - RADIUS < 0:
            self.vx = -self.vx # + random.uniform(0,1)
            #if self.vx >= 5:
            #    self.vx = random.uniform(0,MAX_SPEED)
            print('boundary')

        if self.x + RADIUS > WIDTH:
            self.vx = -self.vx # - random.uniform(0,2)
            #if self.vx <= -5:
            #    self.vx = random.uniform(-MAX_SPEED,0)
            print('boundary')

        if self.y - RADIUS < 0:
            self.vy = -self.vy # + random.uniform(0,2)
            #if self.vy >= 5:
            #    self.vy = random.uniform(0,MAX_SPEED)
            print('boundary')

        if self.y + RADIUS > HEIGHT:
            self.vy = -self.vy # - random.uniform(0,2)
            #if self.vy <= -5:
            #    self.vy = random.uniform(-MAX_SPEED,0)
            print('boundary')

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)

# This object is the point that derives its movement information from a neural network.
class NN_Point:
    def __init__(self):
        self.x = WIDTH/2 #random.uniform(RADIUS, WIDTH-RADIUS)
        self.y = HEIGHT/2 #random.uniform(RADIUS, HEIGHT-RADIUS)
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.color = NN_POINT_COLOR
        self.fitness = 0
        self.brain = Neural_Network.NeuralNetwork()
        self.alive = True

    def update(self):

        move = self.brain.predict(np.random.rand(1,8))

        # update position
        self.x += 10*move[0][0]
        self.y += 10*move[0][1]

        if self.x -RADIUS < 0 or self.x + RADIUS > WIDTH:
            self.alive = False
            print('Point left boundary')
        
        if self.y - RADIUS < 0 or self.y + RADIUS > HEIGHT:
            self.alive = False
            print('Point left boundary')

        # check for collision with walls and adjust behavior
        # if particle interacts with wall do not it proceed past wall
        """
        if self.x - RADIUS < 0:
            self.x = RADIUS

        if self.x + RADIUS > WIDTH:
            self.x = WIDTH - RADIUS

        if self.y - RADIUS < 0:
            self.y = RADIUS

        if self.y + RADIUS > HEIGHT:
            self.y = HEIGHT - RADIUS
        """
    def reset_position(self):
        self.x = WIDTH/2
        self.y = HEIGHT/2

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), RADIUS)


# Check all parameters related to simulation prior to running
# load previous best preformer
# print stats from maintained log 
def start_sequence():
    if os.path.isfile(BEST_NETWORK):
        pass



def main():

    start_sequence()

    pool = Network_Pool()

    while True:
    # iterate through population and run simulation
        for i in range(len(pool.population)):

            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            clock = pygame.time.Clock()

            # Start time
            start_time = pygame.time.get_ticks()   

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
                elapsed_time = pygame.time.get_ticks() - start_time
                seconds = elapsed_time // 1000


                # update and draw blue points
                for point in collision_points:
                    point.update()
                    point.draw(screen)

                    # check for collision with red point
                    distance = np.hypot(point.x - nn_point.x, point.y - nn_point.y)
                    if distance < 2 * RADIUS:
                        print(f"Blue point collided with red point at ({point.x}, {point.y})")
                        # track the neural networks performance
                        nn_point.fitness = elapsed_time
                        running = False

                nn_point.update()
                nn_point.draw(screen)

                # check if point has gone out of bounds.
                # terminate iteration if true
                if nn_point.alive == False:
                    print('dead point')
                    running = False

                pygame.display.flip()
                clock.tick(60)
        
            print('game end')
            pygame.quit()

        pool.order_by_performance()
        pool.save_best()
        pool.population[0].brain = pool.crossover(pool.population[0].brain,pool.population[1].brain)
        pool.mutate_population(pool.rattle(pool.population[0].brain,POPULATION_SIZE))
        pool.reset_population_attributes()


if __name__ == "__main__":
    main()