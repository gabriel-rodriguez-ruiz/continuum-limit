#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 09:16:35 2025

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
min_samples = 1000
tol = 1e-6
theta = np.pi/2
phi_angle = np.pi/2
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
boundary_width = 0.1
boundary_refinement = 10

# def Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu):
#     "Returns the Energy without spin-orbit coupling."
#     chi_k = -2*w_0*(np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y)) - mu
#     return 1/2*( s*B + s_j * np.sqrt(chi_k**2 + Delta**2) -2*w_0*(np.sin(k_x)*np.sin(phi_x) + np.sin(phi_y)*np.sin(phi_y)))

# def f(k_x, k_y):
#     "Returns the Energy without spin-orbit coupling."
#     return Energy(k_x, k_y, phi_x, phi_y, s, s_j, B, Delta, w_0, mu)

# def g(k_x, k_y):
#     return f(k_x, k_y)

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

# def get_energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_s, w_S,
#                mu, Delta_s, Delta_S, B_x,
#                B_y, B_x_S, B_y_S, Lambda, w_1):
#     energies = np.zeros((len(k_x_values), len(k_y_values),
#                         len(phi_x_values), len(phi_y_values), 8))
#     for i, k_x in enumerate(k_x_values):
#         for j, k_y in enumerate(k_y_values):
#             for k, phi_x in enumerate(phi_x_values):
#                 for l, phi_y in enumerate(phi_y_values):
#                     H = get_Hamiltonian_two_layers(k_x, k_y, phi_x, phi_y, w_s, w_S,
#                                         mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
#                                         Lambda, w_1)
#                     E = np.linalg.eigvalsh(H)
#                     for m in range(8):
#                         energies[i, j, k, l, m] = E[m]
#     return energies

def get_energy(k_x, k_y, phi_x, phi_y, w_s, w_S,
               mu, Delta_s, Delta_S, B_x,
               B_y, B_x_S, B_y_S, Lambda, w_1):
    energies = np.zeros(8)
    H = get_Hamiltonian_two_layers(k_x, k_y, phi_x, phi_y, w_s, w_S,
                        mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                        Lambda, w_1)
    energies = np.linalg.eigvalsh(H)
    return energies


def implicit_adaptive_integral_improved(f, g, bounds, tol=1e-6, root_finding_tol=1e-8):
    """
    Adaptive integration that handles multiple roots in the implicit function
    
    Parameters:
    - f: implicit function defining domain (f(x,y) <= 0)
    - g: function to integrate
    - bounds: ((xmin, xmax), (ymin, ymax))
    - tol: integration tolerance
    - root_finding_tol: tolerance for root finding
    
    Returns:
    - integral value
    - error estimate
    """
    (xmin, xmax), (y_bounds) = bounds
    
    def find_all_roots(y, x_guess=None):
        """Find all roots of f(x,y)=0 for given y"""
        roots = []
        
        # Sample function to find potential brackets
        x_samples = np.linspace(xmin, xmax, 1000)
        f_samples = [f(x, y) for x in x_samples]
        
        # Find sign changes that indicate roots
        sign_changes = np.where(np.diff(np.sign(f_samples)))[0]
        
        # Refine each root
        for i in sign_changes:
            try:
                sol = root_scalar(lambda x: f(x, y),
                                bracket=[x_samples[i], x_samples[i+1]],
                                xtol=root_finding_tol)
                if sol.converged:
                    roots.append(sol.root)
            except:
                continue
                
        # Add optional guess-based root if provided
        if x_guess is not None:
            try:
                sol = root_scalar(lambda x: f(x, y),
                                x0=x_guess,
                                xtol=root_finding_tol)
                if sol.converged and xmin <= sol.root <= xmax:
                    roots.append(sol.root)
            except:
                pass
                
        # Remove duplicates and sort
        roots = sorted(list(set(roots)))
        return roots
    
    def integrand_for_y(y):
        """Integrate g(x,y) over x for fixed y, handling multiple regions"""
        roots = find_all_roots(y)
        
        # No roots case
        if not roots:
            # Check if entire interval is inside (f <= 0)
            if f((xmin+xmax)/2, y) <= 0:
                return quad(lambda x: g(x, y), xmin, xmax, epsabs=tol/100)[0]
            else:
                return 0.0
        
        # Determine which regions are inside (f <= 0)
        test_points = [xmin] + roots + [xmax]
        inside = []
        for i in range(len(test_points)-1):
            x_test = (test_points[i] + test_points[i+1])/2
            inside.append(f(x_test, y) <= 0)
        
        # Integrate over all inside regions
        integral = 0.0
        for i, is_inside in enumerate(inside):
            if is_inside:
                a = test_points[i]
                b = test_points[i+1]
                integral += quad(lambda x: g(x, y), a, b, epsabs=tol/100)[0]
        
        return integral
    
    # Perform outer integration over y
    return quad(integrand_for_y, y_bounds[0], y_bounds[1], epsabs=tol)



def implicit_adaptive_integral_with_determinant(f, g, h, bounds, num_points=1000,
                                                tol=1e-6, root_finding_tol=1e-8,
                                                boundary_width=0.1, boundary_refinement=10):
    """
    Adaptive integration that handles multiple roots in the implicit function
    
    Parameters:
    - f: implicit function defining domain (f(x,y) <= 0)
    - g: function to integrate
    - h: implicit function defining domain (h(x,y) = det(H(x,y)) = 0)
    - bounds: ((xmin, xmax), (ymin, ymax))
    - tol: integration tolerance
    - root_finding_tol: tolerance for root finding
    
    Returns:
    - integral value
    - error estimate
    """
    (xmin, xmax), (y_bounds) = bounds
    

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
    
    def integrate_region(a, b, y, is_boundary=False):
        """Integrate with refined settings in boundary regions"""
        if is_boundary:
            # Use higher precision for boundary regions
            subdiv = max(3, int((b - a)/boundary_width * boundary_refinement))
            x_points = np.linspace(a, b, subdiv + 1)
            integral = 0
            for i in range(subdiv):
                integral += quad(lambda x: g(x, y), 
                               x_points[i], x_points[i+1],
                               epsabs=tol/boundary_refinement)[0]
            return integral
        else:
            # Standard integration away from boundaries
            return quad(lambda x: g(x, y), a, b, epsabs=tol)[0]
    def integrand_for_y(y):
        """Integrate g(x,y) over x for fixed y with boundary refinement"""
        boundary_regions = find_boundary_regions(y, f, h)
        
        # No boundary case
        if not boundary_regions:
            if f((xmin+xmax)/2, y) <= 0:
                return integrate_region(xmin, xmax, y)
            return 0.0
        
        # Determine all regions to integrate
        regions = []
        current = xmin
        
        for b_start, b_end in boundary_regions:
            if current < b_start:
                regions.append((current, b_start, False))  # Non-boundary
            regions.append((b_start, b_end, True))  # Boundary
            current = b_end
        
        if current < xmax:
            regions.append((current, xmax, False))  # Final non-boundary
        
        # Integrate all regions
        integral = 0.0
        for a, b, is_boundary in regions:
            # Check if region is inside domain
            test_point = (a + b)/2
            if f(test_point, y) <= 0:
                integral += integrate_region(a, b, y, is_boundary)
        
        return integral
    
    # Perform outer integration over y
    return quad(integrand_for_y, y_bounds[0], y_bounds[1], epsabs=tol)


# def implicit_adaptive_integral_multimin(f, g, bounds, tol=1e-6, root_finding_tol=1e-8, min_samples=50):
#     """
#     Handles multiple local minima that might touch zero without crossing.
    
#     Parameters:
#     - f: implicit function defining domain (f(x,y) <= 0)
#     - g: function to integrate
#     - bounds: ((xmin, xmax), (ymin, ymax))
#     - tol: integration tolerance
#     - root_finding_tol: tolerance for root/minimum finding
#     - min_samples: minimum samples for initial scanning
    
#     Returns:
#     - integral value
#     - error estimate
#     """
#     (xmin, xmax), (y_bounds) = bounds
    
#     def find_all_critical_points(y):
#         """Find all potential boundary points for given y"""
#         results = set()
        
#         # Function for this specific y value
#         def f_y(x):
#             return f(x, y)
        
#         # 1. Initial high-resolution scan to find potential regions
#         x_scan = np.linspace(xmin, xmax, min_samples)
#         f_scan = np.array([f_y(x) for x in x_scan])
#         abs_f_scan = np.abs(f_scan)
        
#         # 2. Find all local minima of |f(x,y)|
#         minima_indices, _ = find_peaks(-abs_f_scan)
#         for idx in minima_indices:
#             if abs_f_scan[idx] < 10*root_finding_tol:  # Near-zero minimum
#                 # Refine using minimization
#                 try:
#                     res = minimize_scalar(
#                         lambda x: abs(f_y(x)),
#                         bounds=(max(xmin, x_scan[max(0, idx-1)]),
#                                  min(xmax, x_scan[min(len(x_scan)-1, idx+1)])),
#                         method='bounded',
#                         xatol=root_finding_tol
#                     )
#                     if res.success and abs(res.fun) < root_finding_tol:
#                         results.add(res.x)
#                 except:
#                     continue
        
#         # 3. Find all sign changes (traditional roots)
#         sign_changes = np.where(np.diff(np.sign(f_scan)))[0]
#         for idx in sign_changes:
#             try:
#                 root = root_scalar(
#                     f_y,
#                     bracket=[x_scan[idx], x_scan[idx+1]],
#                     xtol=root_finding_tol
#                 )
#                 if root.converged:
#                     results.add(root.root)
#             except:
#                 continue
        
#         # 4. Check endpoints
#         for x in [xmin, xmax]:
#             if abs(f_y(x)) < root_finding_tol:
#                 results.add(x)
        
#         return sorted(results)
    
#     def classify_region(a, b, y):
#         """Determine if region between a and b is inside domain"""
#         # Test multiple points including endpoints
#         test_points = np.linspace(a, b, 5)
#         f_values = [f(x, y) for x in test_points]
        
#         # Region is inside if all points <= tolerance
#         is_inside = all(fx <= root_finding_tol for fx in f_values)
        
#         # Special case: all points exactly zero (flat boundary)
#         is_zero_region = all(abs(fx) < root_finding_tol for fx in f_values)
        
#         return is_inside or is_zero_region
    
#     def integrand_for_y(y):
#         """Integrate g(x,y) over x for fixed y"""
#         critical_points = find_all_critical_points(y)
        
#         # No critical points case
#         if not critical_points:
#             # Check if entire interval is inside
#             if f((xmin+xmax)/2, y) <= 0:
#                 return quad(lambda x: g(x, y), xmin, xmax, epsabs=tol/100)[0]
#             return 0.0
        
#         # Add endpoints if needed
#         points = [xmin] + critical_points + [xmax]
#         points = sorted(list(set(points)))  # Remove duplicates
        
#         integral = 0.0
#         for i in range(len(points)-1):
#             a, b = points[i], points[i+1]
#             if classify_region(a, b, y):
#                 # For very small regions, use midpoint rule
#                 # if (b - a) < 10*root_finding_tol:
#                 if False:
#                     integral += g((a+b)/2, y) * (b - a)
#                 else:
#                     integral += quad(lambda x: g(x, y), a, b, epsabs=tol/100)[0]
        
#         return integral
    
#     # Perform outer integration over y
#     return quad(integrand_for_y, y_bounds[0], y_bounds[1], epsabs=tol)


def get_fundamental_energy_with_SOC(phi_x_values, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D, tol=1e-6, root_finding_tol=1e-8, num_points=1000,
                                    boundary_width=0.1, boundary_refinement=10):
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
    fundamental_energy = np.zeros(len(phi_x_values))
    fundamental_energy_error = np.zeros(len(phi_x_values))
    for i, phi_x in enumerate(phi_x_values):
        partial_fundamental_energy = 0
        partial_error = 0
        for j in range(4):
            # print(j)
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
            result, error = implicit_adaptive_integral_with_determinant(f, g, h, bounds, num_points, tol, root_finding_tol, boundary_width, boundary_refinement)
            partial_fundamental_energy += result
            partial_error += error
        fundamental_energy[i] = partial_fundamental_energy
        fundamental_energy_error[i] = partial_error
    return fundamental_energy, fundamental_energy_error

def get_fundamental_energy_two_layers(phi_x_values, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                      Lambda, w_1, E_0, tol=1e-6, root_finding_tol=1e-8, num_points=1000):
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
    fundamental_energy = np.zeros(len(phi_x_values))
    fundamental_energy_error = np.zeros(len(phi_x_values))
    for i, phi_x in enumerate(phi_x_values):
        partial_fundamental_energy = 0
        partial_error = 0
        for j in range(8):
            def wrapped_function_conditional(x, y):
                if isinstance(x, (float, int)):
                    return  get_energy(x, y, phi_x, phi_y, w_s, w_S,
                                   mu, Delta_s, Delta_S, B_x,
                                   B_y, B_x_S, B_y_S, Lambda, w_1)[j]
                elif isinstance(x, (np.ndarray, list)):
                    # Apply the function element-wise for arrays
                    return np.array([get_energy(xi, y, phi_x, phi_y, w_s, w_S,
                                   mu, Delta_s, Delta_S, B_x,
                                   B_y, B_x_S, B_y_S, Lambda, w_1)[j] for xi in x])
                else:
                    raise TypeError("Inputs must be floats or arrays of compatible types.")
            f = wrapped_function_conditional
            g = f
            def h(k_x, k_y):
                return np.linalg.det(get_Hamiltonian_two_layers(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                    Lambda, w_1, E_0))
            # result, error = implicit_adaptive_integral_improved(f, g, bounds, tol=1e-6, root_finding_tol=1e-8)
            result, error = implicit_adaptive_integral_with_determinant(f, g, h, bounds, num_points, tol, root_finding_tol, boundary_width, boundary_refinement)
            partial_fundamental_energy += result
            partial_error += error
        fundamental_energy[i] = partial_fundamental_energy
        fundamental_energy_error[i] = partial_error
    return fundamental_energy, fundamental_energy_error

def get_fundamental_energy_two_layers_for_a_given_phi(phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                      Lambda, w_1, E_0, tol=1e-6, root_finding_tol=1e-8, num_points=1000):
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
    fundamental_energy_error = np.zeros(len(phi_x_values))
    fundamental_energy = 0
    fundamental_energy_error = 0
    for j in range(8):
        def wrapped_function_conditional(x, y):
            if isinstance(x, (float, int)):
                return  get_energy(x, y, phi_x, phi_y, w_s, w_S,
                               mu, Delta_s, Delta_S, B_x,
                               B_y, B_x_S, B_y_S, Lambda, w_1)[j]
            elif isinstance(x, (np.ndarray, list)):
                # Apply the function element-wise for arrays
                return np.array([get_energy(xi, y, phi_x, phi_y, w_s, w_S,
                               mu, Delta_s, Delta_S, B_x,
                               B_y, B_x_S, B_y_S, Lambda, w_1)[j] for xi in x])
            else:
                raise TypeError("Inputs must be floats or arrays of compatible types.")
        f = wrapped_function_conditional
        g = f
        def h(k_x, k_y):
            return np.linalg.det(get_Hamiltonian_two_layers(k_x, k_y, phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                Lambda, w_1, E_0))
        # result, error = implicit_adaptive_integral_improved(f, g, bounds, tol=1e-6, root_finding_tol=1e-8)
        partial_result, partial_error = implicit_adaptive_integral_with_determinant(f, g, h, bounds, num_points, tol, root_finding_tol, boundary_width, boundary_refinement)
        fundamental_energy += partial_result
        fundamental_energy_error += partial_error
    return fundamental_energy, fundamental_energy_error

if __name__ == "__main__":
    start_time = time.time()
    #E, error = get_fundamental_energy_with_SOC(phi_x_values, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D, tol=1e-6, root_finding_tol=1e-8, num_points=1000)
    Lambda = 0
    phi_x = 0
    w_s = 10
    w_S = 1
    mu = -35
    Delta_s = 0.08
    Delta_S = 0.2
    theta = np.pi/2
    phi_angle = np.pi/2
    B_x = B * np.sin(theta) * np.cos(phi_angle)
    B_y = B * np.sin(theta) * np.sin(phi_angle)
    B_x_S = 0
    B_y_S = 0
    w_1 = 1
    E, error = get_fundamental_energy_two_layers(phi_x, phi_y, w_s, w_S, mu, Delta_s, Delta_S, B_x, B_y, B_x_S, B_y_S,
                                           Lambda, w_1, tol=1e-6, root_finding_tol=1e-8, num_points=1000)
    stop_time = time.time()
    elpased_time = stop_time - start_time
    # print(f"Computed area: {E:.8f} ± {error:.2e}")
