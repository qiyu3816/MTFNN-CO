# MTFNN-CO

This repo contains the official implementation for the paper 
[Computation Offloading in Multi-Access Edge Computing: A Multi-Task Learning Approach](https://ieeexplore.ieee.org/document/9079564) by B. Yang, X. Cao, J. Bassey, X. Li and L. Qian. Published in IEEE TMC 2020.

There also contains a work Multi-head Ensemble Multi-Task Learning (MEMTL) as extension. 

## Dataset introduction

The 6 attributes of each MU：

| x1                                                 | x2                                             | x3                        | x4                 | x5       | x6       |
| -------------------------------------------------- | ---------------------------------------------- | ------------------------- | ------------------ | -------- | -------- |
| the amount of input data necessary to be processed | total number of CPU cycles required to process | local CPU cycle frequency | channel power gain | alpha    | beta     |
| ~U(0, 5e5)                                         | =x1*3e3                                        | ~U(0, 1e9)                | ~U(0, 1)           | ~U(0, 1) | =1-alpha |

Preset experimental parameters：

| Name | Value      | Meaning       |
| ----------- | ----------- | ---------------- |
| F_t         | 2.5e9       | total available computing resource on the server |
| kappa       | 1e-28       | parameter for local energy consumption |
| Pt          | 0.3         | transmission power |
| PI          | 0.1         | execution power |
| theta       | 1.0 (second) | the maximum tolerable delay |
| B           | 10e5        | the operational frequency band |
| N0(sigma^2) | 7.96159e-13 | for SINR calculation |