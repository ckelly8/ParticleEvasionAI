import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        #flatten the input layer first
        self.flatten = tf.keras.layers.Flatten(input_shape=(1,5,5))

        #input layer defines NN points 'vision'
        self.input_layer = tf.keras.layers.Dense(25, activation='relu')
        
        #simple 1 hidden layer
        self.hidden_layer1 = tf.keras.layers.Dense(100,activation='relu')
    
        #output is change in velocity for x and y dimensions
        self.output_layer = tf.keras.layers.Dense(2, activation='tanh')
    
    def call(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.output_layer(x)
        return x