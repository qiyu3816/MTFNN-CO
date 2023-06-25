import numpy as np

def over_all_cost(env_params, X, y1, y2):
    """
    Calculate overall_cost in terms of the specified raw features and 2 groups of strategies.
    :param env_params: a 5-element tuple of F, kappa, P_t, P_I, P_d
    :param X: raw features
    :param y1: prediction 1
    :param y2: prediction 2
    :return: y1's overall_cost, y2's overall_cost, locally process's overall_cost, offloading process's overall_cost
    """
    costs_1, costs_2 = [], []
    costs_local, costs_offload = [], []
    mu_num = len(y1[0])

    F = env_params[0]
    kappa = env_params[1]
    P_t = env_params[2]
    P_I = env_params[3]
    P_d = env_params[4]
    for i in range(X.shape[0]):
        cost_1 = 0
        cost_2 = 0
        cost_local = 0
        cost_offload = 0
        for j in range(mu_num):
            s_n = X[i][6 * j]
            w_n = s_n
            c_n = X[i][6 * j + 1]
            f_n = X[i][6 * j + 2]
            r_u = X[i][6 * j + 3]
            r_d = r_u
            alpha = X[i][6 * j + 4]
            if y1[i][j] > 0.1:
                cost_1 += alpha * (s_n / r_u + c_n / (F * y1[i][j]) + w_n / r_d) \
                          + (1 - alpha) * (P_t * s_n / r_u + P_I * c_n / (F * y1[i][j]) + P_d * w_n / r_d)
            else:
                cost_1 += alpha * (c_n / f_n) + (1 - alpha) * (kappa * (f_n ** 2) * c_n)
            if y2[i][j] > 0.1:
                cost_2 += alpha * (s_n / r_u + c_n / (F * y2[i][j]) + w_n / r_d) \
                          + (1 - alpha) * (P_t * s_n / r_u + P_I * c_n / (F * y2[i][j]) + P_d * w_n / r_d)
            else:
                cost_2 += alpha * (c_n / f_n) + (1 - alpha) * (kappa * (f_n ** 2) * c_n)
            cost_local += alpha * (c_n / f_n) + (1 - alpha) * (kappa * (f_n ** 2) * c_n)
            cost_offload += alpha * (s_n / r_u + c_n / (F / mu_num) + w_n / r_d) \
                            + (1 - alpha) * (P_t * s_n / r_u + P_I * c_n / (F / mu_num) + P_d * w_n / r_d)
        costs_1.append(cost_1)
        costs_2.append(cost_2)
        costs_local.append(cost_local)
        costs_offload.append(cost_offload)
    avg_cost_1, avg_cost_2, avg_cost_local, avg_cost_offload = np.mean(costs_1), np.mean(costs_2), np.mean(costs_local), np.mean(costs_offload)
    return avg_cost_1, avg_cost_2, avg_cost_local, avg_cost_offload