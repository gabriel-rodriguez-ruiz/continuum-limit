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
from scipy.signal import find_peaks

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# data_folder = Path(r"/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Data_19_06_25/Data")
data_folder = Path(r"./Data")

# file_to_open = data_folder / "fundamental_energy_phi_x_in_(-0.01,0.01))_B_in_(0.0784-0.082)_Delta=0.08_lambda_R=0_points=16_Nphi=50_Lambda_R=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=16_beta=1000_T=False_phi_angle=1.57_h=0.01_quad.npz"
# file_to_open = data_folder / "/home/gabriel/Python/Dresselhaus/Data/fundamental_energy_phi_x_in_(-0.01,0.01))_B_in_(0.0-0.16)_Delta=0.08_lambda_R=0_points=16_Nphi=50_Lambda_R=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=16_beta=1000_T=False_phi_angle=1.57_h=0.01_quad.npz"
# file_to_open = data_folder / "fundamental_energy_phi_x_in_(-0.01,0.01))_B_in_(0.07200000000000001-0.088)_Delta=0.08_lambda_R=0_points=16_Nphi=100_Lambda_R=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=16_beta=1000_T=False_phi_angle=1.57_h=0.01_quad.npz"
file_to_open = data_folder / "fundamental_energy_phi_x_in_(-0.01,0.01))_B_in_(0.07200000000000001-0.088)_Delta=0.08_lambda_R=0_points=16_Nphi=100_Lambda_R=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=16_beta=1000_T=False_phi_angle=1.57_h=0.01_quad.npz"

Data = np.load(file_to_open)

fundamental_energy = Data["Energy_phi"]
phi_x_values = Data["phi_x_values"]
# phi_x_values = np.linspace(-0.02, 0.02, 50)
B_values = Data["B_values"]
mu = Data["mu"]
# phi_angle = Data["phi_angle"]
# theta = Data["theta"]
Delta = Data["Delta"]
# Delta = 0.08
# L_x = Data["L_x"]
# L_y = Data["L_y"]
# Lambda_R = Data["Lambda_R"]
w_0 = Data["w_0"]
# h = 2e-4
h = Data["h"]
# Nphi = 30
Nphi = Data["Nphi"]

from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

def cuartic_function(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

def parabola(x, c, b, a):
    return a*x**2 + b*x + c

superfluid_density = np.zeros_like(B_values)
superfluid_density_error = np.zeros_like(B_values)

superfluid_density_polinomial = np.zeros_like(B_values)
n_s_xx = np.zeros_like(B_values)

# phi_x_values_fit = np.zeros_like(B_values)
# phi_x_values_fit = np.stack([[slice(20, 30) for _ in range(15)]
#                              + [slice(23, 27)]
#                              + [slice(23, 27)]
#                              + [slice(19, 31)]
#                              + [slice(22, 28) for _ in range(14)]
#                              ])
phi_x_values_fit = np.stack([[slice(20, 30) for _ in range(16)]])

initial_parameters = [1e6, 1e3, 1e7]
fig, ax = plt.subplots()
for i, B in enumerate(B_values):
    # popt, pcov = curve_fit(parabola, phi_x_values,
    #                        fundamental_energy[i, :],
    #                        p0=initial_parameters)
    p = Polynomial.fit(phi_x_values[phi_x_values_fit[0,i]], fundamental_energy[i, phi_x_values_fit[0,i]], 2)
    ax.plot(phi_x_values, fundamental_energy[i, :], label=r"$B/\Delta=$"
            + f"{np.round(B/Delta, 5)}")
    # ax.plot(phi_x_values, fundamental_energy[i, :])
    # ax.plot(phi_x_values, parabola(phi_x_values, *popt), "r--")
    ax.plot(p.linspace()[0], p.linspace()[1], "b--")
    # superfluid_density[i] = popt[2]
    superfluid_density_polinomial[i] = p.convert().coef[2]#/(2*(w_0*L_x*L_y))
    # superfluid_density_error[i] = np.sqrt(np.diag(pcov))[2]/(2*(w_0*L_x*L_y))
    # h = 0.001
    # n_s_xx[i] = ( fundamental_energy[i, -1] - 2*fundamental_energy[i,25] + fundamental_energy[i,0]) / h**2
# ax.plot(phi_x_values, fundamental_energy_2)

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

fig, ax = plt.subplots()
# ax.errorbar(B_values/Delta_0, superfluid_density,yerr=superfluid_density_error,
#             marker="o",
#             markersize=3)
ax.plot(B_values/Delta, superfluid_density_polinomial, "-ob")
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$n_s(B)$")
# ax.plot(B_values, n_s_xx, "-or")
# ax.set_title("Superfluid density "
#           + "\n"
#           + r" $\mu=$" + f"{mu}"
#           + r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
#           + r"; $\theta=$" + f"{np.round(theta, 2)}"
#           + "\n"
#           + r"$L_x=$" + f"{L_x}"
#           + r"; $L_y=$" + f"{L_y}"
#           + r"; $\lambda_R=$" + f"{Lambda_R}"
#           + f"; h={h}"
#           + r"; $N_\varphi=$" + f"{Nphi}")


#%% Find non zero momentum minima of energy

superfluid_density_polinomial = np.zeros_like(B_values)
n_s_xx = np.zeros_like(B_values)
initial_parameters = [1e6, 1e3, 1e7]


def find_local_minima(y):
    """Find local minima in a 1D array."""
    minima = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            minima.append(i)
    return np.array(minima)

N_fit = 4

fig, ax = plt.subplots()
for i, B in enumerate(B_values):
    ax.plot(phi_x_values, fundamental_energy[i, :], label=r"$B/\Delta=$"
            + f"{np.round(B/Delta, 5)}")
    plt.show()
    minima_indices = find_local_minima(fundamental_energy[i, :])
    minima_indices = np.min(minima_indices)
    ax.scatter(phi_x_values[minima_indices], fundamental_energy[i, minima_indices])
    phi_x_values_fit = slice(minima_indices-N_fit, minima_indices+N_fit)
    p = Polynomial.fit(phi_x_values[phi_x_values_fit], fundamental_energy[i, phi_x_values_fit], 2)
    ax.plot(p.linspace()[0], p.linspace()[1], "b--")
    superfluid_density_polinomial[i] = p.convert().coef[2]#/(2*(w_0*L_x*L_y))
ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_0(\phi_x)$")
ax.legend()

fig, ax = plt.subplots()
ax.plot(B_values/Delta, superfluid_density_polinomial, "-ob")
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$n_s(B)$")