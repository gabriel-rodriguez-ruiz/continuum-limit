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
from pauli_matrices import (tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x)

L_x = 500#1000
L_y = L_x
h = 1e-3
n_cores = 16
points = 1 * n_cores
Lambda = 0
w_s = 5/2
w_S = 1/2
mu = -7
Delta_s = 0#0.08
Delta_S = 0.2
theta = np.pi/2
phi_angle = np.pi/2
B_x_S = 0
B_y_S = 0
w_1 = 0.5
E_0 = -6
beta = 1000
T = False
params = {"w_s": w_s, "w_S": w_S,
          "mu": mu, "L_x": L_x, "L_y": L_y,
          "h": h, "Lambda": Lambda, "points": points, "Delta_s": Delta_s,
          "Delta_S": Delta_S, "theta": theta, "phi_angle": phi_angle 
          }

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
        # + Delta_s*np.kron(tau_x, sigma_0)
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

def GetSumOfPositiveEnergies(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1, E_0):
    positive_energy = []
    for m in range(8):
        energy = np.linalg.eigvalsh(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                            Lambda, w_1, E_0))[m]  #1/2 because HBdG not considered in the coefficients
        if energy>0:
            positive_energy.append(energy)
    return np.sum(np.array(positive_energy))

def Fermi_function_efficiently(energy, beta, T=False):
    if T==False:
        return np.zeros_like(energy)
    else:
        if energy <= 0:
            Fermi = 1 / (np.exp(beta*energy) + 1)
        else:
            Fermi = np.exp(-beta*energy) / (1 + np.exp(-beta*energy))
            return Fermi

def get_superconducting_density_efficiently(L_x, L_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                    Lambda, w_1, E_0):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    phi_x_values = [-h, 0, h]
    phi_y_values = [-h, 0, h]
    fundamental_energy = np.zeros((3, 3))
    for k, phi_x in enumerate(phi_x_values):
        for l, phi_y in enumerate(phi_y_values):
            for i, k_x in enumerate(k_x_values):
                for j, k_y in enumerate(k_y_values):
                    positive_energy = GetSumOfPositiveEnergies(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                        Lambda, w_1, E_0)
                    fundamental_energy[k, l] += -1/2*positive_energy + positive_energy * Fermi_function_efficiently(positive_energy, beta, T)
    n_s_xx = 1/(L_x*L_y) * ( fundamental_energy[2,1] - 2*fundamental_energy[1,1] + fundamental_energy[0,1]) / h**2
    n_s_yy = 1/(L_x*L_y) * ( fundamental_energy[1,2] - 2*fundamental_energy[1,1] + fundamental_energy[1,0]) / h**2
    n_s_xy = 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[2,0] - fundamental_energy[0,2] + fundamental_energy[0,0]) / (2*h)**2  #Finite difference of mixed derivatives
    n_s_yx = 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[0,2] - fundamental_energy[2,0] + fundamental_energy[0,0]) / (2*h)**2
    return n_s_xx, n_s_yy, n_s_xy, n_s_yx

def integrate(B):
    n = np.zeros(4)
    B_x = B * np.sin(theta) * np.cos(phi_angle)
    B_y = B * np.sin(theta) * np.sin(phi_angle)
    n[0], n[1], n[2], n[3] = get_superconducting_density_efficiently(L_x, L_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                        Lambda, w_1, E_0)
    return n

if __name__ == "__main__":
    B_values = np.linspace(1.5*Delta_S, 3*Delta_S, points)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    data_folder = Path("Data/")
    name = f"superfluid_density_in_the_lattice_L_x={L_x}_B_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_Delta_s={Delta_s}_Delta_S={Delta_S}_lambda={np.round(Lambda, 2)}_points={points}_theta={np.round(theta,2)}_points={points}_phi_angle={np.round(phi_angle,2)}_h={h}_w_s={w_s}_w_S={w_S}_E_0={E_0}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, n_B_y=n_B_y, B_values=B_values, **params)
    print("\007")
