"""
MTL example
Read the yaml configuration file and take turns training and testing the MTL model using different data sets.
"""

import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score

import time
import yaml

from model.MTL import MTL
from utils.dataset import read_dataset
from utils.loss_function import mtl_custom_loss
from utils.model_op import store_model

with open('../config/MTL_config.yaml', 'r', encoding='utf-8') as f:
    loader = yaml.load(f.read(), Loader=yaml.FullLoader)
file_paths = loader['datasetPath']
model_paths = loader['modelPath']

for i in range(len(file_paths)):

    X_train, X_test, Y_train_class, Y_train_reg, Y_test_class, Y_test_reg \
        = read_dataset(filepath=file_paths[i])

    mu_num = int(X_train.shape[1] / 6)
    print("-----------------------------\nmu_num={}, train_size={}, test_size={}"
          .format(mu_num, X_train.shape[0], X_test.shape[0]))

    model = MTL(mu_num)
    model.compile(loss=mtl_custom_loss, optimizer='adam')

    start = time.time()
    model.fit(X_train, Y_train_reg, epochs=30, verbose=1)
    end = time.time()
    print("[MTL_example] Training cost {}s, 30 epochs".format(end - start))

    start = time.time()
    prediction = model.predict(X_test)
    end = time.time()
    print("[MTL_example] Prediction cost {}ms per sample".format((end - start) * 1000 / X_test.shape[0]))

    mse = K.sum(K.square(prediction - Y_test_reg))
    pre_decision = np.where(prediction < 0.1, 0, 1)
    pre_class = pre_decision.dot(1 << np.arange(pre_decision.shape[-1] - 1, -1, -1))
    print("[MTL_example] mse/accuracy = {}/{}".format(mse / X_test.shape[0], accuracy_score(Y_test_class, pre_class)))

    store_model(model, model_paths[i])
