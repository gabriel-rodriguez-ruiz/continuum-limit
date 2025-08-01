#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:09:24 2025

@author: gabriel
"""

import time
import numpy as np
from analytic_energy import GetAnalyticEnergies
from Adaptive_1D_Integration_with_Implicit_Bounds import get_Hamiltonian

w_0 = 10
mu = -3.49*w_0
Delta = 0.08
B_x = 2*Delta
B_y = 0
Lambda_R = 0
Lambda_D = 0

k_y_values = np.linspace(-np.pi, np.pi, 10000)

start_time = time.time()
for k_y in k_y_values:
    E = GetAnalyticEnergies(0, k_y, 0, 0, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)
end_time = time.time()
elapsed_time_analytic = end_time - start_time

start_time = time.time()
for k_y in k_y_values:
    E = np.linalg.eigvalsh(get_Hamiltonian(0, k_y, 0, 0, w_0, mu, Delta, B_x, B_y, Lambda_R))
end_time = time.time()
elapsed_time_numerical = end_time - start_time