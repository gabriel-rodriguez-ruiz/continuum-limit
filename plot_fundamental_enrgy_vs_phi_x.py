#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:05:28 2024

@author: gabriel
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

data_folder = Path(r"./Data")

# file_to_open = data_folder / "fundamental_energy_phi_x_in_(-0.01,0.01))_B=0.032_Delta_s=0_Delta_S=0.2_lambda=0_points=20_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=20_phi_angle=1.57_h=0.01_w_s=2.5_w_S=0.5_E_0=-6_epsabs=1e-08.npz"
file_to_open = data_folder / "fundamental_energy_phi_x_in_(-0.1,0.1))_B=0.016_Delta_s=0_Delta_S=0.2_lambda=0_points=20_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=20_phi_angle=1.57_h=0.1_w_s=2.5_w_S=0.5_E_0=-6.npz"

Data = np.load(file_to_open)

fundamental_energy = Data["Energy_phi"]
phi_x_values = Data["phi_x_values"]
# phi_x_values = np.linspace(-0.02, 0.02, 50)
B = Data["B"]
mu = Data["mu"]
# phi_angle = Data["phi_angle"]
# theta = Data["theta"]
# Delta = Data["Delta"]
Delta = 0.2
# L_x = Data["L_x"]
# L_y = Data["L_y"]
# Lambda_R = Data["Lambda_R"]
# w_0 = Data["w_0"]
# h = 2e-4
h = Data["h"]
# Nphi = 30
# Nphi = Data["Nphi"]
# epsabs = Data["epsabs"]

from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

def cuartic_function(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

def parabola(x, c, b, a):
    return a*x**2 + b*x + c

def second_derivative(x, y, i):
    """Compute second derivative at index i using central differences."""
    h = x[1] - x[0]  # Assumes uniform spacing
    if i == 0 or i == len(y) - 1:
        raise ValueError("Cannot compute second derivative at boundary points.")
    return (y[i+1] - 2*y[i] + y[i-1]) / (h ** 2)

superfluid_density = np.zeros_like(phi_x_values)
superfluid_density_error = np.zeros_like(phi_x_values)

superfluid_density_polinomial = np.zeros_like(phi_x_values)
n_s_xx = np.zeros_like(phi_x_values)
n_s_xx_global = np.zeros_like(phi_x_values)

# phi_x_values_fit = np.zeros_like(B_values)
# phi_x_values_fit = np.stack([[slice(20, 30) for _ in range(15)]
#                              + [slice(23, 27)]
#                              + [slice(23, 27)]
#                              + [slice(19, 31)]
#                              + [slice(22, 28) for _ in range(14)]
#                              ])
phi_x_values_fit = np.stack([[slice(0, 50) for _ in range(80)]])

initial_parameters = [1e6, 1e3, 1e7]
fig, ax = plt.subplots()
ax.plot(phi_x_values, fundamental_energy, label=r"$B/\Delta=$"
        + f"{np.round(B/Delta, 5)}")
            #         + f"{np.round(B/Delta, 5)}")
ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_0(\phi_x)$")
# plt.legend()
# plt.title("Fundamental energy "
#           + r"$E_0(\phi) = \sum_{\epsilon_k(\phi) < 0} \epsilon_k(\phi)$"
#           + "\n"
#           + r" $\mu=$" + f"{mu}"
#           + r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
#           + r"; $\theta=$" + f"{np.round(theta, 2)}"
#           + "\n"
#           + r"$L_x=$" + f"{L_x}"
#           + r"; $L_y=$" + f"{L_y}"
#           + r"; $\lambda_R=$" + f"{Lambda_R}")
ax.set_box_aspect(2)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
