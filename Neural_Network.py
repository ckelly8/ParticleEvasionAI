import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #input layer defines NN points 'vision'
        self.input_layer = tf.keras.layers.Dense(25, activation='relu',input_shape=(5,))
        #simple 2 layer wide neural network
        self.hidden_layer1 = tf.keras.layers.Dense(50,activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(25,activation='relu')
        self.hidden_layer3 = tf.keras.layers.Dense(5,activation='relu')
        #output is change in velocity for x and y dimensions
        self.output_layer = tf.keras.layers.Dense(2, activation='tanh')
    
    def call(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x