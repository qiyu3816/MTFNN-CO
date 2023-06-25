import tensorflow as tf
from tensorflow.keras import Model

class MEMTL(Model):
    """
    An extensive work for the article
    "Computation Offloading in Multi-Access Edge Computing: A Multi-Task Learning Approach" 2021.
    Multi-head Ensemble Multi-Task Learning.
    """

    def __init__(self, mu_num, head_num):
        """
        Initialize a MEMTL model with specified mu_num and head_num.
        :param mu_num: Number of mobile users.
        :param head_num: Number of prediction heads.
        """
        assert head_num > 1
        super(MEMTL, self).__init__()

        self.mu_num = mu_num
        self.head_num = head_num

        self.d1 = tf.keras.layers.Dense(60, input_dim=mu_num * 6, activation='relu', trainable=False,
                                        kernel_initializer='he_uniform', bias_initializer='he_uniform')
        self.d2 = tf.keras.layers.Dense(10 * mu_num, activation='relu', trainable=False,
                                        kernel_initializer='random_uniform', bias_initializer='random_uniform')
        self.heads_1 = []
        for i in range(self.head_num):
            if i == 0:
                self.heads_1.append(tf.keras.layers.Dense(40, activation='relu'))
            else:
                self.heads_1.append(tf.keras.layers.Dense(40, activation='relu', trainable=False))

        self.heads_2 = []
        for i in range(self.head_num):
            if i == 0:
                self.heads_2.append(tf.keras.layers.Dense(self.mu_num, activation='sigmoid'))
            else:
                self.heads_2.append(tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=False))

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x1s = []
        for layer in self.heads_1:
            x1s.append(layer(x))
        outputs = []
        for i in range(self.head_num):
            outputs.append(self.heads_2[i](x1s[i]))
        return outputs

    def set_backbone(self, trainable):
        """
        Set the backbone to participate in training.
        :param trainable: whether to train
        :return:
        """
        self.d1.trainable = trainable
        self.d2.trainable = trainable

    def set_trainable(self, i):
        """
        Set the i th prediction head to participate in training.
        :param i: the index of the target prediction head
        :return:
        """
        assert i < self.head_num, '[MEMTL] The index of the head to be reset must less than head_num({})'.format(self.head_num)
        for j in range(self.head_num):
            if j == i:
                self.heads_1[j].trainable = True
                self.heads_2[j].trainable = True
            else:
                self.heads_1[j].trainable = False
                self.heads_2[j].trainable = False

    def add_head(self):
        """
        Add a new prediction head.
        :return:
        """
        self.head_num += 1
        self.heads_1.append(tf.keras.layers.Dense(40, activation='relu', trainable=False))
        self.heads_2.append(tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=False))

    def copy_add_head(self, i):
        """
        Copy the i th prediction head and add the copy.
        :param i: the index of the copied prediction head
        :return:
        """
        self.head_num += 1
        self.heads_1.append(tf.keras.layers.Dense(40, activation='relu', trainable=False))
        self.heads_2.append(tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=False))
        self.call(tf.keras.Input(shape=(self.mu_num * 6)))
        self.heads_1[-1].set_weights(self.heads_1[i].get_weights())
        self.heads_2[-1].set_weights(self.heads_2[i].get_weights())

    def reset_heads(self):
        """
        Reinitialize all prediction heads.
        :return:
        """
        for i in range(self.head_num):
            self.heads_1[i] = tf.keras.layers.Dense(40, activation='relu')
            self.heads_2[i] = tf.keras.layers.Dense(self.mu_num, activation='sigmoid')

    def reset_head(self, i):
        """
        Reinitialize the specified prediction head.
        :param i: the index of the target prediction head
        :return:
        """
        assert i < self.head_num, '[MEMTL] The index of the head to be reset must less than head_num({})'.format(self.head_num)
        self.heads_1[i] = tf.keras.layers.Dense(40, activation='relu')
        self.heads_2[i] = tf.keras.layers.Dense(self.mu_num, activation='sigmoid')

    def get_config(self):
        return {"mu_num": self.mu_num, "head_num": self.head_num}
