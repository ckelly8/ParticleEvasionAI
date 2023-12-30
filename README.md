# ParticleEvasionAI

A proof of concept demonstrating the training of a simple neural network in a reinforcement setting.

This project was a test to study more about implementing neural networks using a genetic algorithm. Having had some exposure to genetic algorithms previously, I was always fascinated with how a mostly stochastic approach to machine learning could produce desirable results. 

### Program Paradigm
The following program logic is used for this simulation:
1. An initial population is created. This population is comprised of many neural networks.
2. Each neural network in the population is then placed in it's own simulation. The goal of the neural network is to dodge particles and avoid the edges of the simulation space.
3. Once every neural network particle fails i.e. collides with wall or particle, the typical rules of genetic algorithms are then applied, including crossover and mutation.
4. This repeats infinitely as a convergence is approached.

### Program Quirks
When running this program there are a few things to be aware of.

If changing the neural network architecture, make sure to delete the weights contained in the /Weights folder as well as the Parameters.txt file. These are used for tracking the population progress and depend on the neural network architecture to function properly.

In it's current implementation, convergence is approached very slowly. This could be easily modified by changing the rattle mutation values, however, since the approach taken here is very random it would be easy to miss any form of convergence by using large mutation values.

### Program Visualization
To visualize this program, run the program as you would any python program with the additional command line argument 'visualize' as shown below:
'Python.exe Particle_Dodge.py visualize'

Note that visualization does not display a historic simulation but is instead a method to simulate the best performing neural network in real time. These simulations will continue infinitely.



Have fun and thanks for looking!
