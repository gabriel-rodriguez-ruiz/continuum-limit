#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:30:22 2025

@author: gabriel
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.signal import find_peaks
from analytic_energy import GetAnalyticEnergies
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import time

Delta = 0.08    
B = 1.1*Delta
w_0 = 10
mu = -3.49*w_0
Lambda_R = 0.56  #   0.56
Lambda_D = 0
n_cores = 16
points = 1 * n_cores
phi_x_values = [0]
phi_y = 0
root_finding_tol = 1e-8
num_points = 1000
tol = 1e-6
theta = np.pi/2
phi_angle = np.pi/2
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
boundary_width = 0.1
boundary_refinement = 10

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

def f(k_x)

def h(k_x, k_y):
    return np.linalg.det(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R))


def find_all_roots(phi_x):
    """Find all roots of h(x,y)=det(H(x,y))=0 for given y"""
    roots = []
    
    # Sample function to find potential brackets
    x_samples = np.linspace(-np.pi, np.pi, num_points)
    det_values = [np.real(f(x, 0)) for x in x_samples]
    
    # Find sign changes that indicate roots
    sign_changes = np.where(np.diff(np.sign(det_values)))[0]
    
    # Refine each root
    for i in sign_changes:
        try:
            sol = root_scalar(lambda x: np.real(f(x, 0)),
                            bracket=[x_samples[i], x_samples[i+1]],
                            method='brentq',
                            xtol=root_finding_tol
                            )
            if sol.converged:
                roots.append(sol.root)
        except:
            continue

    #Special case for B>Delta without SOC
    # if np.sqrt(B_x**2 + B_y**2)>=Delta and Lambda_R==0:
    def f_y(x):
        return h(x, y)
    abs_f_scan = np.abs(det_values)
    minima_indices, _ = find_peaks(-abs_f_scan)
    for idx in minima_indices:
        if abs_f_scan[idx] < 1e5*root_finding_tol:  # Near-zero minimum
            # Refine using minimization
            try:
                res = minimize_scalar(
                    lambda x: abs(f_y(x)),
                    bounds=(max(xmin, x_samples[max(0, idx-1)]),
                             min(xmax, x_samples[min(len(x_samples)-1, idx+1)])),
                    method='bounded',
                    tol=root_finding_tol
                )
                if res.success and abs(res.fun) < 1e5*root_finding_tol:
                    roots.append(res.x)
            except:
                continue        # Remove duplicates and sort
    roots = sorted(list(set(roots)))

    return roots