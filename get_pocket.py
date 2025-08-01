#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:34:43 2025

@author: gabriel
"""

import numpy as np
from scipy.optimize import broyden1
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path
from scipy.optimize import minimize_scalar, root_scalar
from analytic_energy import GetAnalyticEnergies
from scipy.signal import find_peaks
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import time

Delta = 0.08    
B = 1.01*Delta
w_0 = 10
mu = -3.49*w_0
Lambda_R = 0.56#0.56#0.56
Lambda_D = 0
n_cores = 16
points = 1 * n_cores
phi_x = 0
phi_y = 0
root_finding_tol = 1e-8
min_samples = 2000
theta = np.pi/2
phi_angle = np.pi/2
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)

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

bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
(xmin, xmax), (y_bounds) = bounds

# f = lambda k_x, k_y: Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu)
# g = f


def find_all_roots(y, f, min_samples):
    """Find all roots of h(x,y)=det(H(x,y))=0 for given y"""
    roots = []
    
    # Sample function to find potential brackets
    x_samples = np.linspace(xmin, xmax, min_samples)
    det_values = [np.real(f(x, y)) for x in x_samples]
    
    # Find sign changes that indicate roots
    sign_changes = np.where(np.diff(np.sign(det_values)))[0]
    
    # Refine each root
    for i in sign_changes:
        try:
            sol = root_scalar(lambda x: np.real(f(x, y)),
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
        if abs_f_scan[idx] < 1e6*root_finding_tol:  # Near-zero minimum
            # Refine using minimization
            try:
                res = minimize_scalar(
                    lambda x: abs(f_y(x)),
                    bounds=(max(xmin, x_samples[max(0, idx-1)]),
                             min(xmax, x_samples[min(len(x_samples)-1, idx+1)])),
                    method='bounded',
                    tol=root_finding_tol
                )
                if res.success and abs(res.fun) < 1e6*root_finding_tol:
                    roots.append(res.x)
            except:
                continue        # Remove duplicates and sort
    roots = sorted(list(set(roots)))

    return roots

def find_all_critical_points(y):
     """Find all potential boundary points for given y"""
     results = set()
     
     # Function for this specific y value
     def f_y(x):
         return f(x, y)
     
     # 1. Initial high-resolution scan to find potential regions
     x_scan = np.linspace(xmin, xmax, min_samples)
     f_scan = np.array([f_y(x) for x in x_scan])
     abs_f_scan = np.abs(f_scan)
     
     # 2. Find all local minima of |f(x,y)|
     minima_indices, _ = find_peaks(-abs_f_scan)
     for idx in minima_indices:
         if abs_f_scan[idx] < 10*root_finding_tol:  # Near-zero minimum
             # Refine using minimization
             try:
                 res = minimize_scalar(
                     lambda x: abs(f_y(x)),
                     bounds=(max(xmin, x_scan[max(0, idx-1)]),
                              min(xmax, x_scan[min(len(x_scan)-1, idx+1)])),
                     method='bounded',
                     tol=root_finding_tol
                 )
                 if res.success and abs(res.fun) < root_finding_tol:
                     results.add(res.x)
             except:
                 continue
     
     # 3. Find all sign changes (traditional roots)
     sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
     for idx in sign_changes:
         try:
             root = root_scalar(
                 f_y,
                 bracket=[x_scan[idx], x_scan[idx+1]],
                 xtol=root_finding_tol
             )
             if root.converged:
                 results.add(root.root)
         except:
             continue
     
     # 4. Check endpoints
     for x in [xmin, xmax]:
         if abs(f_y(x)) < root_finding_tol:
             results.add(x)
     
     return sorted(results)


k_x_values = np.linspace(-np.pi, np.pi)
k_y_values = np.linspace(-np.pi, np.pi, 100)

# roots = find_all_roots(k_y, x_guess=None)

fig, ax = plt.subplots()
for k_y in k_y_values:
    for j in range(4):
        def wrapped_function_conditional(x, y):
            if isinstance(x, (float, int)):
                return GetAnalyticEnergies(x, y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)[j]
            elif isinstance(x, (np.ndarray, list)):
                # Apply the function element-wise for arrays
                return np.array([GetAnalyticEnergies(xi, y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)[j] for xi in x])
            else:
                raise TypeError("Inputs must be floats or arrays of compatible types.")
        f = wrapped_function_conditional
        def h(k_x, k_y):
            return np.linalg.det(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R))
        roots = find_all_roots(k_y, f, min_samples)
        # critical_points = find_all_critical_points(k_y)
        # ax.scatter(critical_points, k_y*np.ones_like(critical_points), color="blue")
        ax.scatter(roots, k_y*np.ones_like(roots), color="blue")
end_time = time.time()
        # ax.scatter(k_y*np.ones_like(critical_points), roots, color="red")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_aspect('equal', 'box')
