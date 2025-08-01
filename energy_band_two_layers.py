#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:04:02 2025

@author: gabriel
"""

import numpy as np
from pauli_matrices import (tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x, sigma_z)
import matplotlib.pyplot as plt
from analytic_energy import get_cuartic_solution
from scipy.integrate import quad, dblquad
from scipy.optimize import root_scalar
import time

phi_x = 0
phi_y = 0
g_xx = 1
g_yy = 1
g_xy = 0
g_yx = 0
Lambda = 0#0.056#0.56
w_s = 5/2
w_S = 1/2
mu = -7#-3.5*w_S #-3*w_s
Delta_s = 0 #0.08
Delta_S = 0.2#0.2
theta = np.pi/2
phi_angle = np.pi/2
w_1 = 0.5
k_y = 0 
N_k_x = 1000
# k_y_values = np.linspace(0.7, 0.8, N_k_y)
k_x_values = np.linspace(-np.pi, np.pi, N_k_x)
theta = np.pi/2
phi_angle = 0   #np.pi/2
B = 0.1  #0.032#0.004     #4*Delta_S#*Delta_S
g_s = 2
g_S = 1
B_x_S = g_S * B * np.sin(theta) * np.cos(phi_angle)
B_y_S = g_S * B * np.sin(theta) * np.sin(phi_angle)
B_x = g_s * B * np.sin(theta) * np.cos(phi_angle)
B_y = g_s * B * np.sin(theta) * np.sin(phi_angle)
E_0 = -6
B_s = B*g_s
B_S = B*g_S

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1, E_0):
    r""" A semiconductor plane over a superconductor plane. The semiconductor
    has spin-orbit coupling and magnetic field.
    
    .. math::
        H_\mathbf{k} = \frac{1}{2} (H_s + H_S + H_{w_1})
        
        H_s = -2w_s\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right)
        \tau_0\sigma_0
        -\mu\tau_z\sigma_0
        + 2\lambda\left(\sin(k_x)\cos(\Phi_x)\tau_z\sigma_y
        + \cos(k_x)\sin(\Phi_x)\tau_0\sigma_y
        - \sin(k_y)\cos(\Phi_y)\tau_z\sigma_x
        - \sin(k_y)\sin(\Phi_y)\tau_0\sigma_x
        - B_x\tau_0\sigma_x - B_y\tau_0\sigma_y \right)
        
        H_S = -2w_S\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \tau_z\sigma_0
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right)
        \tau_0\sigma_0
        (E_0-\mu)\tau_z\sigma_0
        + \Delta \tau_x\sigma_0
        
        H_{w_1} = -w_1 \alpha_x\tau_z\sigma_0
            
    """
    H_s = (
        -2*w_s*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta_s*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_S = (
        -2*w_S*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) + (E_0- mu) * np.kron(tau_z, sigma_0)
        # + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
        #             + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
        #             - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
        #             - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x_S*np.kron(tau_0, sigma_x) - B_y_S*np.kron(tau_0, sigma_y)
        + Delta_S*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_w_1 = -w_1 * np.kron(tau_z, sigma_0)
    H = np.block([
            [H_s, H_w_1],
            [H_w_1, H_S]
        ])
    return H

def get_Hamiltonian_without_SOC(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_s, B_S,
                     w_1, E_0):
    r""" A semiconductor plane over a superconductor plane. The semiconductor
    has spin-orbit coupling and magnetic field.
    
    .. math::
        H_\mathbf{k} = 1/2(H_s + H_S + H_{w_1})
        
        H_s = -2w_s\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \sigma_z
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right) \sigma_0)
       
        -\mu\sigma_z
        - B \sigma_0
        
        H_S = -2w_S\left(\cos(k_x)\cos(\Phi_x) + \cos(k_y)\cos(\Phi_y)\right)
        \sigma_z
        - \left(\sin(k_x)\sin(\Phi_x) + \sin(k_y)\sin(\Phi_y)\right)
        \sigma_0 + 
        (E_0-\mu)\sigma_z
        -B_S \sigma_0
        + \Delta \sigma_x
        
        H_{w_1} = -w_1 \alpha_x\sigma_z
            
    """
    H_s = (
        -2*w_s*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * sigma_z
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * sigma_0) 
        - mu * sigma_z
        - B_s * sigma_0
        + Delta_s * sigma_x
            ) * 1/2
    H_S = (
        -2*w_S*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * sigma_z
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * sigma_0
               ) + (E_0-mu) * sigma_z
        - B_S * sigma_0
        + Delta_S * sigma_x
            ) * 1/2
    H_w_1 = -w_1 * sigma_z
    H = np.block([
            [H_s, H_w_1],
            [H_w_1, H_S]
        ])
    return H

def get_coefficient_array(a, b, c, d, e, f, g):
    """
    H = 1/2 * (a K[0, 3] + b K[0, 0] + c K[0, 1] + 
            d K[3, 3] + e K[3, 0] + f K[3, 1] + g K[1, 3] )
    
    returns np.array[[c0, c1, c2, c3, c4]]
    """
    # Array convertida a Python
    array = np.array([
            1/16 * (a**4 + b**4 + c**4 + 2 * c**2 * d**2 + d**4 - 2 * c**2 * e**2 - 2 * d**2 * e**2 + e**4
                    + 8 * b * c * e * f - 2 * c**2 * f**2 + 2 * d**2 * f**2 - 2 * e**2 * f**2 + f**4
                    + 8 * a * d * (b * e - c * f)
                    + 2 * (c**2 + d**2 + e**2 - f**2) * g**2 + g**4
                    - 2 * a**2 * (b**2 - c**2 + d**2 + e**2 - f**2 + g**2)
                    - 2 * b**2 * (c**2 + d**2 + e**2 + f**2 + g**2)
                    ),
            1/2 * (a**2 * b - b**3 - 2 * a * d * e - 2 * c * e * f + b * (c**2 + d**2 + e**2 + f**2 + g**2)
                ),
            1/2 * (-a**2 + 3 * b**2 - c**2 - d**2 - e**2 - f**2 - g**2
                ),
            -2 * b,
            1
    ])
    return array

def GetAnalyticEnergies(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_S, B_s, B_S,
                     w_1, E_0):
    """
    H = 1/2 * (A K[0, 3] + B K[0, 0] + C K[0, 1] + 
            D K[3, 3] + E K[3, 0] + F K[3, 1] + G K[1, 3] )
    """
    energies = np.zeros(4)
    A = -(w_s + w_S) * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y)) + E_0/2 - mu
    B = (w_s + w_S) * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y)) - (B_s + B_S)/2
    C = Delta_S/2
    D = -(w_s - w_S) * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y)) - E_0/2
    E = (w_s - w_S) * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y)) - (B_s - B_S)/2
    F = - Delta_S/2
    G = -2 * w_1
    coefficient_array = get_coefficient_array(A, B, C, D, E, F, G)      
    c_0, c_1, c_2, c_3, c_4 = coefficient_array
    for m in range(4):
        energies[m] = np.real(get_cuartic_solution(c_0, c_1, c_2, c_3)[m])  #HBdG has already considered the 1/2 in the coefficients
    return energies

def find_roots(n_samples=1000, root_finding_tol=1e-8):
    roots = []
    def f_2(k_x):
        return np.linalg.eigvalsh(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                            Lambda, w_1, E_0))[2]

    def f_3(k_x):
        return np.linalg.eigvalsh(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                            Lambda, w_1, E_0))[3]
    for f in [f_2, f_3]:
        x_samples = np.linspace(-np.pi, np.pi, n_samples)
        det_values = [f(x) for x in x_samples]
        # Find sign changes that indicate roots
        sign_changes = np.where(np.diff(np.sign(det_values)))[0]
        # Refine each root
        for i in sign_changes:
            try:
                sol = root_scalar(lambda x: f(x),
                                bracket=[x_samples[i], x_samples[i+1]],
                                method='brentq',
                                xtol=root_finding_tol
                                )
                if sol.converged:
                    roots.append(sol.root)
            except:
                continue
    return roots

E_k_x = np.zeros((len(k_x_values), 8))
# E_4_by_4 = np.zeros((len(k_y_values), 4))
# E_4_by_4_minus_B = np.zeros((len(k_y_values), 4))
# E_4_by_4_analytic = np.zeros((len(k_y_values), 4))

for i, k_x in enumerate(k_x_values):
    E_k_x[i, :] = np.linalg.eigvalsh(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                        Lambda, w_1, E_0))
    # E_4_by_4[i, :] = np.linalg.eigvalsh(get_Hamiltonian_without_SOC(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_s, B_S,
    #                      w_1, E_0))
    # E_4_by_4_minus_B[i, :] = np.linalg.eigvalsh(get_Hamiltonian_without_SOC(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, -B_s, -B_S,
    #                      w_1, E_0))
    # E_4_by_4_minus_B[len(k_y_values)-i-1, :] = (-E_4_by_4[i, :])
    # E_4_by_4_analytic[i, :] = GetAnalyticEnergies(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_S, B_s, B_S,
    #                      w_1, E_0)
fig, ax = plt.subplots()

for i in range(8):
    ax.plot(k_x_values, E_k_x[:, i])
for j in range(4):
    ax.plot(k_x_values, -np.abs(E_k_x[:, j]), linewidth=3, linestyle="dashed")

# for i in range(4):
#     ax.plot(k_y_values, E_4_by_4[:, i],
#     linewidth=3, linestyle="dashed")
#     ax.plot(k_y_values, E_4_by_4_minus_B[:, i],
#     linewidth=3, linestyle="dashed")
#     ax.scatter(k_y_values, E_4_by_4_analytic[:, i])
ax.scatter(find_roots(n_samples=1000, root_finding_tol=1e-8), 
           np.zeros_like(find_roots(n_samples=1000, root_finding_tol=1e-8)))
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$E(k_x,k_y=0)$")
ax.set_title("Bilayer with " + r"$g_s B/(2\Delta_S)=$"+ f"{1/2*g_s*B/Delta_S}; " + r"$\lambda=$" +
             f"{Lambda}; " + r"$\phi_x=$" + f"{phi_x}")
ax.axhline(0, linestyle="dashed",
           color="black")

def get_Hamiltonian_single_layer(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R):
    """ Periodic Hamiltonian in x and y with flux.
    """
    H = (
        -2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda_R*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2
    return H


def integrand_for_k_y(k_y):
    def g(k_x):
        return np.sum(-np.abs(np.linalg.eigvalsh(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                            Lambda, w_1, E_0)))[:4])
    roots = find_roots()
    integral, error = quad(g, -np.pi, np.pi, points=roots, limit=100)
    return integral

def integrand_for_k_y_single_layer(k_y):
    def g(k_x):
        return np.sum(-np.abs(np.linalg.eigvalsh(get_Hamiltonian_single_layer(k_x, k_y, phi_x, phi_y, w_0=10, mu=mu, Delta=0.2, B_x=B_x_S, B_y=B_y_S, Lambda_R=0)))[:2])
    roots = find_roots()
    integral, error = quad(g, -np.pi, np.pi, points=roots, limit=100)
    return integral

def get_fundamental_energy():
    return quad(integrand_for_k_y, -np.pi, np.pi)

def get_fundamental_energy_double():
    return dblquad(integrand_for_k_y, -np.pi, np.pi)
# if __name__=="__main__":
#     start_time = time.time()
#     print(get_fundamental_energy())
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(elapsed_time)
    
fig, ax = plt.subplots()
k_y_values = np.linspace(-np.pi, np.pi, 50)
integrand = [integrand_for_k_y(k_y) for k_y in k_y_values]
ax.plot(k_y_values, integrand)
# ax.plot(k_y_values, [integrand_for_k_y_single_layer(k_y) for k_y in k_y_values])
ax.set_xlabel(r"$k_y$")
