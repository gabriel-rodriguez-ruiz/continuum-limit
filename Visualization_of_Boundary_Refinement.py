#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 17:05:20 2025

@author: gabriel
"""

import matplotlib.pyplot as plt
from Adaptive_1D_Integration_with_Implicit_Bounds import implicit_adaptive_integral_improved
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.signal import find_peaks
from analytic_energy import GetAnalyticEnergies
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x

y_test = -0.467
Delta = 0.08    
B = 1.1*Delta
w_0 = 10
mu = -3.49*w_0
Lambda_R = 0.56   #0.056  #   0.56
Lambda_D = 0
n_cores = 16
points = 1 * n_cores
phi_x_values = [0]
phi_x = 0
phi_y = 0
j = 1
root_finding_tol = 1e-8
tol = 1e-6
theta = np.pi/2
phi_angle = np.pi/2
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)

bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
(xmin, xmax), (y_bounds) = bounds
num_points = 1000
boundary_width = 0.1
boundary_refinement = 10

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

def find_boundary_regions(y, f, h):
    """Find all boundary regions for given y"""
    # First find all roots
    def find_all_roots():
        """Find all roots of h(x,y)=det(H(x,y))=0 for given y"""
        roots = []
        
        # Sample function to find potential brackets
        x_samples = np.linspace(xmin, xmax, num_points)
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
    
    roots = find_all_roots()
    if not roots:
        return []
    
    # Create boundary intervals
    boundary_intervals = []
    for root in roots:
        left = max(xmin, root - boundary_width/2)
        right = min(xmax, root + boundary_width/2)
        boundary_intervals.append((left, right))
    
    # Merge overlapping intervals
    merged = []
    for interval in sorted(boundary_intervals):
        if not merged:
            merged.append(list(interval))
        else:
            last = merged[-1]
            if interval[0] <= last[1]:  # Overlapping
                last[1] = max(last[1], interval[1])
            else:
                merged.append(list(interval))
    
    return merged

def wrapped_function_conditional(x, y):
    if isinstance(x, (float, int)):
        return GetAnalyticEnergies(x, y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)[j]
    elif isinstance(x, (np.ndarray, list)):
        # Apply the function element-wise for arrays
        return np.array([GetAnalyticEnergies(xi, y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D)[j] for xi in x])
    else:
        raise TypeError("Inputs must be floats or arrays of compatible types.")
f = wrapped_function_conditional
g = f
def h(k_x, k_y):
    return np.linalg.det(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R))
# h = f
boundaries = find_boundary_regions(y_test, f, h)

fig, ax = plt.subplots()

x_plot = np.linspace(-3, 3, 10000)
ax.plot(x_plot, [f(x, y_test) for x in x_plot], label='f(x,y)')
# ax.plot(x_plot, [h(x, y_test) for x in x_plot], label='h(x,y)')

ax.axhline(0, color='gray', linestyle='--')
for b_start, b_end in boundaries:
    ax.axvspan(b_start, b_end, color='red', alpha=0.2, label='Boundary regions')
plt.title(f"Boundary regions at y={y_test}")
plt.xlabel("x")
plt.legend()
plt.show()