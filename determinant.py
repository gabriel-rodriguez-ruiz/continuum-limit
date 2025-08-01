#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 07:41:15 2025

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
from scipy.optimize import root_scalar

Delta = 0.08    
B = 1.1*Delta
w_0 = 10
mu = -3.49*w_0
Lambda_R = 0.56
Lambda_D = 0
n_cores = 16
points = 1 * n_cores
phi_x = 0
phi_y = 0
root_finding_tol = 1e-8
num_points = 2000
theta = np.pi/2
phi_angle = np.pi/2
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
k_y = 0
number_of_k_y_points = 500

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R):
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

def get_Hamiltonian_two_layers(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1):
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
        -\mu\tau_z\sigma_0
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
        # + Delta_s*np.kron(tau_x, sigma_0)
            ) * 1/2
    H_S = (
        -2*w_S*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
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

def matrix_func(param):
    return get_Hamiltonian(param, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R)

# def matrix_func(param):
#     return get_Hamiltonian_two_layers(param, k_y, phi_x=0, phi_y=0, w_s=10, w_S=1, mu=mu, Delta_s=0.8,
#                                       Delta_S=0.2, B_x=B_x, B_y=0, B_x_S=0, B_y_S=0,
#                         Lambda=0, w_1=1)



def find_determinant_zeros(matrix_func, param_range, num_points=1000):
    """Find where determinant crosses zero"""
    params = np.linspace(param_range[0], param_range[1], num_points)
    det_values = [np.real(np.linalg.det(matrix_func(p))) for p in params]
    
    # Find where sign changes occur
    sign_changes = np.where(np.diff(np.sign(det_values)))[0]
    
    # Refine each root using root_scalar
    roots = []
    for i in sign_changes:
        root_result = root_scalar(
            lambda p: np.linalg.det(matrix_func(p)),
            bracket=[params[i], params[i+1]],
            xtol=root_finding_tol,
            method='brentq'
        )
        roots.append(root_result.root)
    
    return roots

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    param_range = [-np.pi, np.pi]
    roots = []
    k_y_values = np.linspace(-np.pi, np.pi, number_of_k_y_points)
    fig, ax = plt.subplots()
    for k_y in k_y_values:
        roots.append(find_determinant_zeros(matrix_func, param_range, num_points))
    end_time = time.time()
    elapsed_time = end_time - start_time
        # ax.plot(roots, k_y*np.ones_like(roots))
    for i, (x_vals, y_vals) in enumerate(zip(roots, k_y_values)):
        if x_vals and y_vals:  # Check if both x and y lists are not empty
            ax.plot(x_vals, y_vals*np.ones_like(x_vals), "or")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_aspect('equal', 'box')
