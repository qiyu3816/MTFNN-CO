import datetime
import yaml

from tqdm import tqdm
import numpy as np
import pandas as pd
from gekko import GEKKO

from utils.dataset_generate import range_random

def co_gen(node_num, sample_num):
    X = np.array([])
    Y = np.array([])

    satisfy_delay_num = 0

    F_t = 806.4e9  # considering multiple cores
    kappa = 1e-28
    P_t = 0.3  # transmission power
    P_I = 0.1
    theta = 2.0
    B = 10e5
    N0 = 7.96159e-13  # xi**2, for SINR calculation

    s_mu, s_sigma, s_low, s_up = 2.025e5, 1e5, 1e5, 4.05e5
    fl_mu, fl_sigma, fl_low, fl_up = 1e9, 5e8, 2e8, 1.6e9

    for _ in tqdm(range(sample_num)):
        s = range_random(s_mu, s_sigma, node_num, s_low, s_up)
        c = s * 3e3  # required cycles
        f_local = range_random(fl_mu, fl_sigma, node_num, fl_low, fl_up)

        alpha = np.random.rand(node_num)
        beta = 1 - alpha

        h = np.random.rand(node_num)  # the channel power gain of node i to the MEC server

        sinr = P_t * (h ** 2) / (N0 + np.sum(P_t * (h ** 2)))
        r_u = B * np.log2(1 + sinr)
        # download process omitted in conv co

        tau_local = c / f_local
        epsilon_local = kappa * (f_local ** 2) * c
        cost_local = alpha * tau_local + beta * epsilon_local

        cost_trans = alpha * s / r_u + beta * P_t * s / r_u
        cost_proc = alpha * c / F_t + beta * P_I * c / F_t

        m = GEKKO()  # Initialize gekko
        m.options.SOLVER = 1  # APOPT is an MINLP solver

        # optional solver settings with APOPT
        m.solver_options = ['minlp_maximum_iterations 100',  # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 10',  # treat minlp as nlp
                            'minlp_as_nlp 0',  # nlp sub-problem max iterations
                            'nlp_maximum_iterations 50',  # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1',  # maximum deviation from whole number
                            'minlp_integer_tol 0.05',  # convergence tolerance
                            'minlp_gap_tol 0.01']

        decision = m.Array(m.Var, node_num, integer=True)
        for dd in decision:
            dd.value = 1
            dd.lower = 0
            dd.upper = 1
        allocation = m.Array(m.Var, node_num)
        for al in allocation:
            al.value = 1 / node_num
            al.lower = 10e-5
            al.upper = 1
        cost_local_param = m.Array(m.Param, node_num)
        for idx, pp in enumerate(cost_local_param):
            pp.value = cost_local[idx]
        cost_trans_param = m.Array(m.Param, node_num)
        for idx, pp in enumerate(cost_trans_param):
            pp.value = cost_trans[idx]
        cost_proc_param = m.Array(m.Param, node_num)
        for idx, pp in enumerate(cost_proc_param):
            pp.value = cost_proc[idx]
        m.Equation(m.sum(decision * allocation) <= 1)
        # Due to the excessively complex gekko expressions, maximum tolerable delay is not considered here.
        m.Obj(m.sum((1 - decision) * cost_local_param + decision * (cost_trans_param + cost_proc_param / allocation)))
        try:
            m.solve(disp=False)
        except Exception as e:
            print("Exception.")
            continue

        x = np.array([])
        # unique features
        for n in range(node_num):
            x = np.append(x, [s[n], c[n], f_local[n], h[n], alpha[n], beta[n]])
        y = np.array([])
        for dd in decision:
            y = np.append(y, [int(dd.VALUE[0])])
        for al in allocation:
            y = np.append(y, [float(al.VALUE[0])])

        latency = (1 - y[:node_num]) * tau_local + y[:node_num] * (s / r_u + s / r_u / y[-node_num:])
        if np.all(latency <= theta):
            satisfy_delay_num += 1

        y[-node_num:] *= y[-2 * node_num:-node_num]
        X = np.append(X, x)
        Y = np.append(Y, y)

    X = np.reshape(X, (len(X) // (node_num * 6), node_num * 6))
    Y = np.reshape(Y, (len(Y) // (node_num * 2), node_num * 2))
    np.set_printoptions(precision=4, suppress=True)
    print(f"{satisfy_delay_num}/{sample_num} satisfy the tolerable delay.")
    # common features
    cmf = np.array([F_t, kappa, P_t, P_I, theta, B, N0, s_mu, s_sigma, s_low, s_up, fl_mu, fl_sigma, fl_low, fl_up])

    return X, Y, cmf


if __name__ == "__main__":

    node_num, sample_num = 3, 1000
    X, Y, cmf = co_gen(node_num, sample_num)

    tag = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    df = pd.DataFrame(np.concatenate((X, Y), axis=1))
    df.to_csv(f"../datasets/testbed/{node_num}nodes_{sample_num}samples_{tag}.csv", header=None, index=False)

    yml_dict = {}
    yml_keys = ['F_t', 'kappa', 'P_t', 'P_I', 'theta', 'B', 'N0', 's_mu', 's_sigma', 's_low', 's_up', 'fl_mu', 'fl_sigma', 'fl_low', 'fl_up']
    for i in range(len(yml_keys)):
        yml_dict[yml_keys[i]] = float(cmf[i])
    yml_path = f'../datasets/testbed/{node_num}nodes_{sample_num}samples_{tag}.yaml'
    with open(yml_path, 'w') as file:
        yaml.dump(yml_dict, file)
