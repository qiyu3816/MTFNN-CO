"""
MEMTL example:
Read the configuration file and repeat experiments to compare the inference performance of
MTFNN and MEMTL with different number of prediction heads under different number of mobile terminals.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score

import time
import yaml

from utils.dataset import split_dataset2subsets_train
from utils.loss_function import custom_loss
from utils.cost_calc import over_all_cost
from utils.model_op import store_model
from model.MEMTL import MEMTL
from model.MTL import MTL

def train_test_store(dataset_path, head_range, env_params, model_path):
    """
    experiment steps:
    1. The data set is divided into 10,000 samples for testing, and the remaining data is used as the training set
    2. Use the training set to conduct bootstrap sampling to obtain the training subsets for the prediction heads
    3. The backbone is trained on all training sets
    4. Reinitialize all prediction heads and freeze the backbone on different subsets for training
    5. Train the MTL using the same training set and use it as a benchmark for comparison
    6. Output experiment results to file
    For ease of calculation and presentation, after the last prediction head training is completed, the overall_cost is calculated and the result is stored in the file
    :param dataset_path: dataset path
    :param head_range: range of prediction head number
    :param env_params: environment parameters to calculate the overall_cost
    :param model_path: the base path for model storing
    :return: string text for test results
    """

    mtl_sign = False
    mtl = None
    memtl = None

    test_res = []
    for head_num in range(head_range[0], head_range[1] + 1):
        Xs, Ys, X_train, Y_train, test_raw = split_dataset2subsets_train(dataset_path, subset_num=head_num + 1)
        mu_num = int(Xs[0].shape[1] / 6)
        print("[train_test_store] mu_num={}, head_num={}, env_size={}, MTL train_size={}"
              .format(mu_num, head_num, Xs[0].shape[0], X_train.shape[0]))
        env_summary = "mu_num={}, head_num={}, env_size={}, MTL train_size={}\n".format(mu_num, head_num, Xs[0].shape[0], X_train.shape[0])

        # MTL initialize and train
        if not mtl_sign:
            mtl_sign = True
            mtl = MTL(mu_num)

            begin = time.time()
            mtl.compile(loss=custom_loss, optimizer='adam')
            end = time.time()
            print("[train_test_store] mtl compile cost {}ms."
                  .format(mu_num, head_num, (end - begin) * 1000))
            begin = time.time()
            mtl.fit(X_train, Y_train[:, -mu_num:], epochs=30, verbose=0)
            end = time.time()
            print("[train_test_store] mtl training cost {}ms.".format((end - begin) * 1000))

        # MEMTL initialize and train
        if head_num == 2:
            memtl = MEMTL(mu_num, head_num)
            memtl.set_backbone(True)
            memtl.set_trainable(0)

            begin = time.time()
            memtl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss)
            end = time.time()
            print("[train_test_store] Init, mu_num={}, head_num={}, compile cost {}ms."
                  .format(mu_num, head_num, (end - begin) * 1000))
            begin = time.time()
            memtl.fit(X_train, Y_train[:, -mu_num:], epochs=30, verbose=0)
            end = time.time()
            print("[train_test_store] Init, train cost {}ms.".format((end - begin) * 1000))
            memtl.reset_head(0)
        else:
            memtl.add_head()

        for i in range(head_num):
            if head_num > 2 and i != head_num - 1:
                continue
            memtl.set_backbone(False)
            memtl.set_trainable(i)
            begin = time.time()
            memtl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss)
            end = time.time()
            print("[train_test_store] Head {}, compile cost {}ms.".format(i + 1, (end - begin) * 1000))

            begin = time.time()
            memtl.fit(Xs[i], Ys[i][:, -mu_num:], epochs=30, verbose=0)
            end = time.time()
            print("[train_test_store] Head {}, train cost {}ms.".format(i + 1, (end - begin) * 1000))


        print("[train_test_store] >>>>>>>>>> Test data {} samples <<<<<<<<<<".format(Xs[-1].shape[0]))
        begin = time.time()
        outputs = memtl.predict(Xs[-1])
        end = time.time()
        print("[train_test_store] Model prediction cost {}ms per sample.".format(
            (end - begin) * 1000 / Xs[-1].shape[0]))

        # The inference results of multiple prediction heads are traversed,
        # the one with the correct offloading decision and the lowest mse is picked first,
        # then the one with the correct offloading decision is picked next,
        # and if there is no correct decision, the one with the lowest mse is picked.
        global_mse = []
        global_decision = []
        global_pick = []
        global_prediction = []
        for j in range(Xs[-1].shape[0]):
            mses = []
            decisions = []
            for ii in range(memtl.head_num):
                prediction = outputs[ii][j]
                mses.append(K.sum(K.square(prediction - Ys[-1][j, - memtl.mu_num:])))
                pre_decision = np.where(prediction < 0.1, 0, 1)
                pre_class = pre_decision.dot(1 << np.arange(pre_decision.shape[-1] - 1, -1, -1))
                decisions.append(bool(pre_class == Ys[-1][j, - (memtl.mu_num + 1)]))
            index = [k for k in range(memtl.head_num)]# if decisions[k]]
            if len(index) > 0:
                picked = index[np.argmin([mses[k] for k in index])]
                global_pick.append(picked)
                global_mse.append(mses[picked])
                global_decision.append(decisions[picked])
                global_prediction.append(outputs[picked][j])
            else:
                picked = np.argmin(mses)
                global_pick.append(picked)
                global_mse.append(mses[picked])
                global_decision.append(decisions[picked])
                global_prediction.append(outputs[picked][j])

        mse = K.sum(global_mse)
        mse = np.array(mse)
        print("[train_test_store] MEMTL prediction mse error={} per sample.".format(mse / Xs[-1].shape[0]))
        right_cnt = sum(d for d in global_decision)
        memtl_summary = "MEMTL prediction cost {:.4f}ms per sample, "\
                            .format((end - begin) * 1000 / Xs[-1].shape[0]) + \
                        str(round(mse / Xs[-1].shape[0], 4)) + "/" + str(round(right_cnt / len(global_decision), 4))
        print("[train_test_store] MEMTL abs_acc={}".format(right_cnt / len(global_decision)))
        for h in range(memtl.head_num):
            print("[train_test_store] MEMTL head {} pick_rate={}".format(h + 1, np.sum(
                    np.array(global_pick) == h) / len(global_pick)))

        begin = time.time()
        prediction = mtl.predict(Xs[-1])
        end = time.time()
        print("[train_test_store] MTL prediction cost {}ms per sample".format((end - begin) * 1000 / Xs[-1].shape[0]))
        mtl_mse = K.sum(K.square(prediction - Ys[-1][:, -mu_num:]))
        pre_decision = np.where(prediction < 0.1, 0, 1)
        pre_class = pre_decision.dot(1 << np.arange(pre_decision.shape[-1] - 1, -1, -1))
        print("[train_test_store] mse/accuracy = {}/{}"
              .format(mtl_mse / Xs[-1].shape[0], accuracy_score(Ys[-1][:, -(mu_num + 1)], pre_class)))

        mtl_summary = "MTL   prediction cost {:.4f}ms per sample, {:.4f}/{:.4f}"\
            .format((end - begin) * 1000 / Xs[-1].shape[0], mtl_mse / Xs[-1].shape[0],
                    accuracy_score(Ys[-1][:, -(mu_num + 1)], pre_class))

        memtl_avg_cost, mtl_avg_cost, _, _ = over_all_cost(env_params, test_raw, global_prediction, prediction)
        memtl_summary = memtl_summary + " overall cost {:.4f}\n".format(memtl_avg_cost)
        mtl_summary = mtl_summary + " overall cost {:.4f}\n".format(mtl_avg_cost)

        print("[train_test_store] overall cost {:.4f}/{:.4f}".format(memtl_avg_cost, mtl_avg_cost))
        if head_num == head_range[1]:
            final_feature_results = np.concatenate((test_raw, np.array(global_prediction)), axis=1)
            final_feature_results = np.concatenate((final_feature_results, np.array(prediction)), axis=1)
            print("[train_test_store] storing test set and results ", final_feature_results.shape)
            df = pd.DataFrame(final_feature_results)
            df.to_csv("../results/" + dataset_path.split('/')[-1][:7] + "results.csv", index=False, header=None)

        test_res.append(env_summary)
        test_res.append(memtl_summary)
        test_res.append(mtl_summary)

        store_path = model_path + "{}h".format(head_num)
        store_model(memtl, store_path)
        print("[train_test_store] MEMTL model stored to", store_path)
    return test_res

def MEMTL_exp():
    """
    Experiments of MEMTLs with different mu_num and head_num specified by configuration.
    :return:
    """
    with open('../config/MEMTL_config.yaml', 'r', encoding='utf-8') as f:
        loader = yaml.load(f.read(), Loader=yaml.FullLoader)

    file_paths = loader['datasetPath']
    head_ranges = loader['headNumRange']
    model_paths = loader['modelPath']
    F = loader['F']
    kappa = loader['kappa']
    P_t = loader['P_t']
    P_I = loader['P_I']
    P_d = loader['P_d']
    env_params = (F, kappa, P_t, P_I, P_d)

    results = []
    for i in range(len(file_paths)):
        dataset = file_paths[i]
        print("########################### MEMTL dataset %s ###########################" % (dataset.split('/')[-1].split('.')[0]))
        results.append(train_test_store(dataset, head_ranges[i], env_params, model_paths[i]))

    file = open(loader['result_file'], 'w')
    for r in results:
        for s in r:
            file.write(s)
    file.close()

if __name__ == "__main__":
    MEMTL_exp()
