import tensorflow as tf
from tensorflow.keras import Model

class MTL(Model):
    """
    Implementation of the article
    "Computation Offloading in Multi-Access Edge Computing: A Multi-Task Learning Approach" 2020.
    """
    def __init__(self, mu_num):
        """
        Initialize a MTFNN model with preset mu_num.
        :param mu_num: Number of mobile users.
        """
        super(MTL, self).__init__()
        self.mu_num = mu_num

        self.d1 = tf.keras.layers.Dense(60, input_dim=mu_num * 6, activation='relu',
                                        kernel_initializer='he_uniform', bias_initializer='he_uniform')
        self.d2 = tf.keras.layers.Dense(20, activation='relu',
                                        kernel_initializer='random_uniform', bias_initializer='random_uniform')
        self.d3 = tf.keras.layers.Dense(40, activation='relu')
        self.d4 = tf.keras.layers.Dense(self.mu_num, activation='sigmoid')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)

    def get_config(self):
        return {"mu_num": self.mu_num}
