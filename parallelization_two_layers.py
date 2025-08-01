#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:34:43 2025

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from Adaptive_1D_Integration_with_Implicit_Bounds import implicit_adaptive_integral_improved, get_fundamental_energy_with_SOC, get_fundamental_energy_two_layers
from pauli_matrices import (tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x, sigma_z)

h = 1e-2#5e-5#1e-3#2e-4
Nphi = 20  #50
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
Lambda = 0
w_s = 5/2
w_S = 1/2
mu = -7
Delta_s = 0#0.08
Delta_S = 0.2
theta = np.pi/2
phi_angle = np.pi/2
g_s = 10
g_S = 1
w_1 = 0.5
boundary_width = 0.001
boundary_refinement = 10
E_0 = -6
B_c = 0.032

params = {"w_s": w_s,
          "mu": mu, "phi_x_values": phi_x_values,
          "h": h, "Nphi":Nphi, "root_finding_tol": root_finding_tol,
          "num_points": num_points, "tol": tol, "Lambda": Lambda,
          "g_S": g_S, "g_s": g_s,
          }

def Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu):
    "Returns the Energy without spin-orbit coupling."
    chi_k = -2*w_0*(np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y)) - mu
    return 1/2*( s*B + s_j * np.sqrt(chi_k**2 + Delta**2) -2*w_0*(np.sin(k_x)*np.sin(phi_x) + np.sin(phi_y)*np.sin(phi_y)))


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

def integrate(B):
    B_x_S = g_S * B * np.sin(theta) * np.cos(phi_angle)
    B_y_S = g_S * B * np.sin(theta) * np.sin(phi_angle)
    B_x = g_s * B * np.sin(theta) * np.cos(phi_angle)
    B_y = g_s * B * np.sin(theta) * np.sin(phi_angle)
    E, error = get_fundamental_energy_two_layers(phi_x_values, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                                 Lambda, w_1, E_0, tol=1e-6, root_finding_tol=1e-8, num_points=1000)
    return E

if __name__ == "__main__":
    B_values = np.linspace(0, 3*B_c, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    Energy_phi = np.array(results_pooled)
    data_folder = Path("Data/")
    name = f"fundamental_energy_phi_x_in_({np.min(phi_x_values)},{np.round(np.max(phi_x_values),3)}))_B_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta_s={Delta_s}_Delta_S={Delta_S}_lambda={np.round(Lambda, 2)}_points={points}_Nphi={Nphi}_g_xx={g_xx}_g_xy={g_xy}_g_yy={g_yy}_g_yx={g_yx}_theta={np.round(theta,2)}_points={points}_phi_angle={np.round(phi_angle,2)}_h={h}_w_s={w_s}_w_S={w_S}_E_0={E_0}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , Energy_phi=Energy_phi, B_values=B_values, **params)
    print("\007")
