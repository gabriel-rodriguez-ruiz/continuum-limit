#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:34:43 2025

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from Adaptive_1D_Integration_with_Implicit_Bounds import implicit_adaptive_integral_improved, get_fundamental_energy_with_SOC

w_0 = 10
mu = -3.49*w_0
Delta = 0.08   
Lambda_R = 0#0.056   #0.56#0.056
Lambda_D = 0
phi_angle = np.pi/2 #np.pi/2 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
mu = -3.49*w_0#-2*t
beta = 1000
T = False
h = 1e-3#1e-3#5e-5#1e-3#2e-4
Nphi = 50  #50
# phi_x_values = np.linspace(-h, h, Nphi)
phi_x_values = np.linspace(-h, h, Nphi)
phi_y = 0
g_xx = 1
g_yy = 1
g_xy = 0
g_yx = 0
N = 1000
n_cores = 16
points = 1 * n_cores
root_finding_tol = 1e-8
tol = 1e-6      #1e-6
num_points = 1000
boundary_width = 0.1
boundary_refinement = 10

params = {"w_0": w_0,
          "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
          "h": h, "Nphi":Nphi, "Lambda_R": Lambda_R, "root_finding_tol": root_finding_tol,
          "num_points": num_points, "tol": tol, "Lambda_R": Lambda_R,
          "Lambda_D": Lambda_D
          }

def Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu):
    "Returns the Energy without spin-orbit coupling."
    chi_k = -2*w_0*(np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y)) - mu
    return 1/2*( s*B + s_j * np.sqrt(chi_k**2 + Delta**2) -2*w_0*(np.sin(k_x)*np.sin(phi_x) + np.sin(phi_y)*np.sin(phi_y)))


def Second_derivative(k_x, k_y, s, s_j, B, Delta, w_0, mu):
    chi_k = -2*w_0*(np.cos(k_x) + np.cos(k_y)) - mu
    return s_j * w_0 * np.cos(k_x) * chi_k / np.sqrt(Delta**2 + chi_k**2)



def get_superfluid_density(k_x_values, k_y_values, B, Delta, w_0, mu):
    superfluid_density = 0
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for s in [-1, 1]:
                for s_j in [-1, 1]:
                    if Energy(k_x, k_y, s, s_j, B, Delta, w_0, mu)<0:
                        superfluid_density += Second_derivative(k_x, k_y, s, s_j, B, Delta, w_0, mu)
    return superfluid_density

def get_fundamental_energy_without_SOC(phi_x_values, phi_y, B, Delta, w_0, mu):
    fundamental_energy = np.zeros(len(phi_x_values))
    fundamental_energy_error = np.zeros(len(phi_x_values))
    for i, phi_x in enumerate(phi_x_values):
        partial_fundamental_energy = 0
        partial_error = 0
        for s in [-1, 1]:
            for s_j in [-1, 1]:
                f = lambda k_x, k_y: Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu)
                g = f
                bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
                result, error = implicit_adaptive_integral_improved(f, g, bounds, tol=1e-6, root_finding_tol=1e-8)
                partial_fundamental_energy += result
                partial_error += error
        fundamental_energy[i] = partial_fundamental_energy
        fundamental_energy_error[i] = partial_error
    return fundamental_energy, fundamental_energy_error

# def get_fundamental_energy_with_Rashba(phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D):
#     fundamental_energy = np.zeros(len(phi_x_values))
#     fundamental_energy_error = np.zeros(len(phi_x_values))
#     for i, phi_x in enumerate(phi_x_values):
#         partial_fundamental_energy = 0
#         partial_error = 0
#         bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
#         # for j in range(4):
#         #     @return_specific_index(j) # Decorate to return the j value (index j)
#         #     def f(k_x, k_y):
#         #         return GetAnalyticEnergies(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)
#         #     g = f
#         # functions = create_2arg_wrappers(GetAnalyticEnergies)
#         for j in range(4):
#             f = lambda k_x, k_y: GetAnalyticEnergies(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)[j]
#             g = f
#             result, error = implicit_adaptive_integral_improved(f, g, bounds, tol=1e-6, root_finding_tol=1e-8)
#             partial_fundamental_energy += result
#             partial_error += error
#         fundamental_energy[i] = partial_fundamental_energy
#         fundamental_energy_error[i] = partial_error
#     return fundamental_energy, fundamental_energy_error

def integrate(B):
    if Lambda_R == 0:
        E, error = get_fundamental_energy_without_SOC(phi_x_values, phi_y, B, Delta, w_0, mu)
    else:
        B_x = B * np.sin(theta) * np.cos(phi_angle)
        B_y = B * np.sin(theta) * np.sin(phi_angle)
        E, error = get_fundamental_energy_with_SOC(phi_x_values, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D,
                                                   tol, root_finding_tol, num_points, boundary_width, boundary_refinement)
    return E

if __name__ == "__main__":
    # B_values = np.linspace(0.95*Delta, 1.05*Delta, points)
    B_values = np.linspace(0.88*Delta, 0.92*Delta, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    Energy_phi = np.array(results_pooled)
    data_folder = Path("Data/")
    name = f"fundamental_energy_phi_x_in_({np.min(phi_x_values)},{np.round(np.max(phi_x_values),3)}))_B_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta={Delta}_lambda_R={np.round(Lambda_R, 2)}_points={points}_Nphi={Nphi}_Lambda_R={Lambda_R}_g_xx={g_xx}_g_xy={g_xy}_g_yy={g_yy}_g_yx={g_yx}_theta={np.round(theta,2)}_points={points}_beta={beta}_T={T}_phi_angle={np.round(phi_angle,2)}_h={h}_quad.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , Energy_phi=Energy_phi, B_values=B_values, **params)
    print("\007")
