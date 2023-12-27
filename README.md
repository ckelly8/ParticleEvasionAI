# ParticleEvasionAI

A proof of concept demonstrating the training of a simple neural network in a reinforcement setting.

This project was a test to study more about implementing neural networks using a genetic algorithm. Having had some exposure to genetic algorithms previously, I was always fascinated with how a mostly stochastic approach to machine learning could produce desirable results. 

This program works under the following paradigm:
- An initial population is created comprised of many neural networks.
- These individual neural networks are then given their own simulation to dodge particles within.
- The typical rules of genetic algorithms are then applied, including crossover and mutation.
- This repeats infinitely as a convergence is approached.

Some quirks should be mentioned if you decide to test this program.
- If changing the neural network architecture, make sure to delete the weights contained in the /Weights folder as well as the Parameters.txt file. These are used for tracking the population progress and depend on the neural network architecture to function properly.
- To visualize this program, simply run the program as you normally with the additional command line argument 'visualize'
- In it's current implementation, convergence is approached very slowly. This could be modified by changing the rattle mutation values, however, since the approach taken here is very random it would be easy to miss any form of convergence by using large mutation values.
