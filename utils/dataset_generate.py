import numpy as np
from tqdm import tqdm
import time

def range_random(mu, sigma, size, lower=None, upper=None):
    """
    Generate a random array of normal distribution.
    :param mu: Mean.
    :param sigma: Variance.
    :param size: Array size.
    :param lower: [optional] Lower bound.
    :param upper: [optional] Upper bound.
    :return Raw array.
    """
    # Generate the initial random array
    arr = np.random.normal(mu, sigma, size)
    if lower is None or upper is None:
        return arr

    # Regenerate elements outside the desired range
    while np.any(arr < lower) or np.any(arr > upper):
        arr[arr < lower] = np.random.normal(mu, sigma, np.sum(arr < lower))
        arr[arr > upper] = np.random.normal(mu, sigma, np.sum(arr > upper))
    return arr

def resource_allocation_gen(D, mode='full', step=0.05):
    """
    Generate all possible allocations based on the given decisions and step.
    """
    # Find the indices where D has a value of 1 (offload)
    indices = np.where(D == 1)[0]
    choices = np.arange(step, 1 + step, step)

    # Generate
    num_combinations = len(choices) ** len(indices)
    arrays = np.zeros((num_combinations, len(D)))

    for i in range(num_combinations):
        for j, idx in enumerate(indices):
            value_idx = (i // (len(choices) ** j)) % len(choices)
            arrays[i, idx] = choices[value_idx]
    # Select the valid allocation
    resource_sum = np.sum(arrays, axis=-1)
    if mode == 'full':
        final_arrays = arrays[np.abs(resource_sum - 1) < 10e-6]
    elif mode == 'part':
        final_arrays = arrays[resource_sum <= 1]
    return final_arrays

def CO_MINLP_GEN(node_num, sample_num):
    """
    Solve the MINLP problem of computation offloading through exhaustive searching.
    :param node_num: Node number.
    :param sample_num: Required sample number.
    :return
    """
    X = np.array([])
    Y = np.array([])

    satisfy_delay_num = 0

    F_t = 7.5e9  # considering multiple cores
    kappa = 1e-28
    P_t = 0.3  # transmission power
    P_I = 0.1
    B = 10e5
    N0 = 7.96159e-13  # xi**2, for SINR calculation
    time_consumption = []
    for i in range(sample_num):

        s = range_random(1e5, 4e4, node_num, 1e4, 2.5e5).astype(int)
        c = s * 3e3  # required cycles
        w = range_random(1e5, 4e4, node_num, 1e4, 2.5e5).astype(int)
        theta = range_random(0.8, 0.15, node_num, 0.01, 1.0).astype(float)
        f_local = range_random(8e8, 4e8, node_num, 0, 1.5e9).astype(int)

        alpha = np.random.rand(node_num)
        beta = 1 - alpha

        h = np.random.rand(node_num)  # the channel power gain of node i to the MEC server

        sinr = P_t * (h ** 2) / (N0 + np.sum(P_t * (h ** 2)))
        r_u = B * np.log2(1 + sinr)
        # download power is generally less than uploading for mobile terminal,
        # but the up-down load is symmetrical, e.i., P_d = P_t, r_d = r_u
        r_d = r_u

        tau_local = c / f_local
        epsilon_local = kappa * (f_local ** 2) * c
        cost_local = alpha * tau_local + beta * epsilon_local

        optimal_cost = float('inf')
        optimal_D = None
        optimal_F = None

        tolerable_cost = float('inf')
        tolerable_D = None
        tolerable_F = None

        D = np.arange(2 ** node_num - 1)
        D_bin = np.zeros(shape=(2 ** node_num - 1, node_num), dtype=int)
        begin = time.time()
        for id in range(len(D)):
            tmp = D[id]
            for j in range(node_num):
                D_bin[id][j] = tmp & 1
                tmp //= 2
            # resource allocation
            Fs = resource_allocation_gen(D_bin[id], 'part')
            for j in range(Fs.shape[0]):
                F = Fs[j]
                F = np.where(D_bin[id] > 0, F, 0.1)
                # tau_offload = s / r_u + c / (F_t * F) + w / r_d
                # epsilon_offload = P_t * s / r_u + P_I * c / (F_t * F) + P_t * w / r_d
                cost_offload = np.where(D_bin[id] > 0, alpha * (s / r_u + c / (F_t * F) + w / r_d)
                                        + beta * (P_t * s / r_u + P_I * c / (F_t * F) + P_t * w / r_d), 0)
                delays = np.where(D_bin[id] > 0, s / r_u + c / (F_t * F) + w / r_d, c / f_local)
                delay_satisfy = np.where(delays < theta, 1, 0)
                F = np.where(D_bin[id] > 0, F, 0)

                cost_total = np.sum((1 - D_bin[id]) * cost_local + D_bin[id] * cost_offload)
                if cost_total < optimal_cost:
                    optimal_cost, optimal_D, optimal_F = cost_total, D_bin[id], F
                if np.alltrue(delay_satisfy == 1):
                    tolerable_cost, tolerable_D, tolerable_F = cost_total, D_bin[id], F
        end = time.time()
        time_consumption.append((end - begin) * 1000)

        x = np.array([])
        # unique features
        for n in range(node_num):
            x = np.append(x, [s[n], c[n], w[n], theta[n], f_local[n], h[n], alpha[n]])
        x = np.append(x, [F_t, kappa, P_t, P_I, B, N0])  # common features
        if tolerable_D is not None:
            satisfy_delay_num += 1
            optimal_cost, optimal_D, optimal_F = tolerable_cost, tolerable_D, tolerable_F
        y = np.concatenate((optimal_D, optimal_F), axis=0)
        y = np.append(y, optimal_cost)
        X = np.append(X, x)
        Y = np.append(Y, y)
    X = np.reshape(X, (len(X) // (node_num * 7 + 6), node_num * 7 + 6))
    Y = np.reshape(Y, (len(Y) // (node_num * 2 + 1), node_num * 2 + 1))
    print(f"{satisfy_delay_num}/{sample_num} satisfy the tolerable delay.")
    print(f"{np.mean(time_consumption)} ms per sample.")
    return X, Y

def CONV_CO_MINLP_GEN(node_num, sample_num):
    """
    Solve the MINLP problem of conventional computation offloading through exhaustive searching.
    :param node_num: Node number.
    :param sample_num: Required sample number.
    :return
    """
    X = np.array([])
    Y = np.array([])

    satisfy_delay_num = 0

    F_t = 2.5e9  # considering multiple cores
    kappa = 1e-28
    P_t = 0.3  # transmission power
    P_I = 0.1
    theta = 1.0
    B = 10e5
    N0 = 7.96159e-13  # xi**2, for SINR calculation
    time_consumption = []
    for i in tqdm(range(sample_num)):

        s = range_random(2.5e5, 5e4, node_num, 0, 5e5).astype(int)
        c = s * 3e3  # required cycles
        f_local = range_random(5.0e8, 2.0e8, node_num, 0, 1e9).astype(int)

        alpha = np.random.rand(node_num)
        beta = 1 - alpha

        h = np.random.rand(node_num)  # the channel power gain of node i to the MEC server

        sinr = P_t * (h ** 2) / (N0 + np.sum(P_t * (h ** 2)))
        r_u = B * np.log2(1 + sinr)
        # no download process in conv co

        tau_local = c / f_local
        epsilon_local = kappa * (f_local ** 2) * c
        cost_local = alpha * tau_local + beta * epsilon_local

        optimal_cost = float('inf')
        optimal_D = None
        optimal_F = None

        tolerable_cost = float('inf')
        tolerable_D = None
        tolerable_F = None

        D = np.arange(2 ** node_num)
        D_bin = np.zeros(shape=(2 ** node_num, node_num), dtype=int)
        begin = time.time()
        for id in range(len(D)):
            if id == 0:  # all local processing
                for j in range(node_num):
                    D_bin[id][j] = 0
                Fs = np.atleast_2d(np.zeros(node_num, dtype=float))
            else:
                tmp = D[id]
                for j in range(node_num):
                    D_bin[id][j] = tmp & 1
                    tmp //= 2
                # resource allocation
                Fs = resource_allocation_gen(D_bin[id], 'full', step=0.02)
            for j in range(Fs.shape[0]):
                F = Fs[j]
                F = np.where(D_bin[id] > 0, F, 0.00001)
                # tau_offload = s / r_u + c / (F_t * F)
                # epsilon_offload = P_t * s / r_u + P_I * c / (F_t * F)
                cost_offload = np.where(D_bin[id] > 0, alpha * (s / r_u + c / (F_t * F))
                                        + beta * (P_t * s / r_u + P_I * c / (F_t * F)), 0)
                delays = np.where(D_bin[id] > 0, s / r_u + c / (F_t * F), c / f_local)
                delay_satisfy = np.where(delays < theta, 1, 0)
                F = np.where(D_bin[id] > 0, F, 0)

                cost_total = np.sum((1 - D_bin[id]) * cost_local + D_bin[id] * cost_offload)
                if cost_total < optimal_cost:
                    optimal_cost, optimal_D, optimal_F = cost_total, D_bin[id], F
                if np.alltrue(delay_satisfy == 1):
                    tolerable_cost, tolerable_D, tolerable_F = cost_total, D_bin[id], F
        end = time.time()
        time_consumption.append((end - begin) * 1000)

        x = np.array([])
        # unique features
        for n in range(node_num):
            x = np.append(x, [s[n], c[n], f_local[n], h[n], alpha[n], beta[n]])
        x = np.append(x, [F_t, kappa, P_t, P_I, theta, B, N0])  # common features
        if tolerable_D is not None:
            satisfy_delay_num += 1
            optimal_cost, optimal_D, optimal_F = tolerable_cost, tolerable_D, tolerable_F
        y = np.concatenate((optimal_D, optimal_F), axis=0)
        y = np.append(y, optimal_cost)
        X = np.append(X, x)
        Y = np.append(Y, y)
    X = np.reshape(X, (len(X) // (node_num * 6 + 7), node_num * 6 + 7))
    Y = np.reshape(Y, (len(Y) // (node_num * 2 + 1), node_num * 2 + 1))
    np.set_printoptions(precision=4, suppress=True)
    print(f"{satisfy_delay_num}/{sample_num} satisfy the tolerable delay.")
    print(f"{np.mean(time_consumption)} ms per sample.")
    return X, Y

