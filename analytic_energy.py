# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:48:57 2024

@author: Gabriel
"""

import sympy as sp
import numpy as np
from sympy.physics.matrices import msigma
from sympy.physics.quantum.tensorproduct import TensorProduct
#from numba import jit, float64, int64

def getAnalyticEnergy(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D):
    A = -2*w_0 * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
    B = -2*w_0 * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
    C = -B_y + Lambda_R * np.cos(k_x)*np.sin(phi_x) + Lambda_D * np.cos(k_y)*np.sin(phi_y)
    D = -B_x - Lambda_R * np.cos(k_y)*np.sin(phi_y) - Lambda_D * np.cos(k_x)*np.sin(phi_x)
    E = Lambda_R * np.sin(k_x)*np.cos(phi_x) + Lambda_D * np.sin(k_y)*np.cos(phi_y)
    F = -Lambda_R * np.sin(k_y)*np.cos(phi_y) - Lambda_D * np.sin(k_x)*np.cos(phi_x) 
    c_0, c_1, c_2, c_3, x, a, b, c, d, e, f, Mu, delta = sp.symbols("c_0 c_1 c_2 c_3 x a b c d e f Mu delta")
    solution = sp.solve(x**4 + c_3*x**3 + c_2*x**2 + c_1*x + c_0, x)
    H = (a * TensorProduct(msigma(3), sp.eye(2)) + b * TensorProduct(sp.eye(2), sp.eye(2))
        - Mu * TensorProduct(msigma(3), sp.eye(2)) + c * TensorProduct(sp.eye(2), msigma(2))
        + d * TensorProduct(sp.eye(2), msigma(1)) + delta * TensorProduct(msigma(1), sp.eye(2))
        + e * TensorProduct(msigma(3), msigma(2)) + f * TensorProduct(msigma(3), msigma(1)) 
        )
    M = H - x * sp.eye(4)
    y = sp.series(M.det(), x, 0, 4)
    d_3 = y.coeff(x, n=3)
    d_2 = y.coeff(x, n=2)
    d_1 = y.coeff(x, n=1)
    d_0 = y.coeff(x, n=0)
    Solution = []
    for i in range(4):
        S = solution[i].args[1].subs([(c_0, d_0), (c_1, d_1), (c_2, d_2), (c_3, d_3)])
        Solution.append(S.subs([(a, A), (b, B), (c, C), (d, D),
                                (e, E), (f, F), (Mu, mu), (delta, Delta)]))
    return np.real([complex(sp.N(Solution[0][0])), complex(sp.N(Solution[1][0])),
            complex(sp.N(Solution[2][0])), complex(sp.N(Solution[3][0]))])

def getAnalyticEnergies(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D):
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values), 4))
    c_0, c_1, c_2, c_3, x, a, b, c, d, e, f, Mu, delta = sp.symbols("c_0 c_1 c_2 c_3 x a b c d e f Mu delta")
    solution = sp.solve(x**4 + c_3*x**3 + c_2*x**2 + c_1*x + c_0, x)
    H = (a * TensorProduct(msigma(3), sp.eye(2)) + b * TensorProduct(sp.eye(2), sp.eye(2))
        - Mu * TensorProduct(msigma(3), sp.eye(2)) + c * TensorProduct(sp.eye(2), msigma(2))
        + d * TensorProduct(sp.eye(2), msigma(1)) + delta * TensorProduct(msigma(1), sp.eye(2))
        + e * TensorProduct(msigma(3), msigma(2)) + f * TensorProduct(msigma(3), msigma(1)) 
        )
    M = H - x * sp.eye(4)
    y = sp.series(M.det(), x, 0, 4)
    d_3 = y.coeff(x, n=3)
    d_2 = y.coeff(x, n=2)
    d_1 = y.coeff(x, n=1)
    d_0 = y.coeff(x, n=0)
    Solution = [0, 0, 0, 0]
    for i in range(4):
        Solution[i] = solution[i].args[1].subs([(c_0, d_0), (c_1, d_1), (c_2, d_2), (c_3, d_3)])
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    A = -2*w_0 * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
                    B = -2*w_0 * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
                    C = -B_y + Lambda_R * np.cos(k_x)*np.sin(phi_x) + Lambda_D * np.cos(k_y)*np.sin(phi_y)
                    D = -B_x - Lambda_R * np.cos(k_y)*np.sin(phi_y) - Lambda_D * np.cos(k_x)*np.sin(phi_x)
                    E = Lambda_R * np.sin(k_x)*np.cos(phi_x) + Lambda_D * np.sin(k_y)*np.cos(phi_y)
                    F = -Lambda_R * np.sin(k_y)*np.cos(phi_y) - Lambda_D * np.sin(k_x)*np.cos(phi_x) 
                    for m in range(4):
                        energies[i, j, k, l, m] = np.real(complex(sp.N(Solution[m].subs([(a, A), (b, B), (c, C), (d, D),
                                                (e, E), (f, F), (Mu, mu), (delta, Delta)])[0])))  
    return energies

#@jit
def get_cuartic_solution(c0, c1, c2, c3):
    # Componentes de la expresión
    term1 = -((2 * c2) / 3)
    term2 = c3**2 / 4
    term3_numerator = 12 * c0 + c2**2 - 3 * c1 * c3
    term3_denominator = (27 * c1**2 - 72 * c0 * c2 + 2 * c2**3 - 9 * c1 * c2 * c3 + 27 * c0 * c3**2) + np.sqrt(
        -4 * (12 * c0 + c2**2 - 3 * c1 * c3)**3 + (27 * c1**2 - 72 * c0 * c2 + 2 * c2**3 - 9 * c1 * c2 * c3 + 27 * c0 * c3**2)**2
        )
    term3 = (2**(1/3) * term3_numerator) / (3 * term3_denominator**(1/3))
    term4 = 1 / (3 * 2**(1/3)) * term3_denominator**(1/3)
    
    sqrt_term1 = np.sqrt(term1 + term2 + term3 + term4)
    
    term5 = -((4 * c2) / 3)
    term6 = c3**2 / 2
    term7 = -(2**(1/3) * term3_numerator) / (3 * term3_denominator**(1/3))
    term8 = -1 / (3 * 2**(1/3)) * term3_denominator**(1/3)
    
    additional_term = (-8 * c1 + 4 * c2 * c3 - c3**3) / (4 * np.sqrt(term1 + term2 + term3 + term4))

    sqrt_term2 = np.sqrt(term5 + term6 + term7 + term8 - additional_term)
    sqrt_term3 = np.sqrt(term5 + term6 + term7 + term8 + additional_term)
    
    result = np.array([-c3 / 4 - 1/2 * sqrt_term1 - 1/2 * sqrt_term2,
              -c3 / 4 - 1/2 * sqrt_term1 + 1/2 * sqrt_term2,
              -c3 / 4 + 1/2 * sqrt_term1 - 1/2 * sqrt_term3,
              -c3 / 4 + 1/2 * sqrt_term1 + 1/2 * sqrt_term3])
    
    return result

#@jit
def get_coefficient_array(a, b, c, d, e, f, Delta, mu):
    """
    H = a K[3, 0] + b K[0, 0] - mu K[3, 0] + c K[0, 2] + 
            d K[0, 1] + Delta K[1, 0] + e K[3, 2] + f K[3, 1]
    """
    # Array convertida a Python
    array = np.array([
        (1j * c + d - 1j * e - f) * (
            -Delta * (-1j * c * Delta + d * Delta - 1j * e * Delta + f * Delta) - (-1j * c + d + 1j * e - f) *
            (-((-1j * c + d - 1j * e + f) * (1j * c + d + 1j * e + f)) + (a + b - mu) ** 2)
        ) + Delta * (a ** 2 * Delta - b ** 2 * Delta - c ** 2 * Delta - d ** 2 * Delta - 2j * d * e * Delta + e ** 2 * Delta +
                    2j * c * f * Delta + f ** 2 * Delta + Delta ** 3 - 2 * a * Delta * mu + Delta * mu ** 2) + (-a + b + mu) *
        ((-((-1j * c + d - 1j * e + f) * (1j * c + d + 1j * e + f)) + (a + b - mu) ** 2) * (-a + b + mu) + Delta * (-a * Delta -
                                                                                                                      b * Delta + Delta * mu)),
        4 * (a ** 2 * b - b ** 3 + b * c ** 2 + b * d ** 2 - 2 * a * c * e + b * e ** 2 - 2 * a * d * f + b * f ** 2 + b * Delta ** 2 - 2 * a * b * mu + 2 * c * e * mu + 2 * d * f * mu + b * mu ** 2),
        -2 * (a ** 2 - 3 * b ** 2 + c ** 2 + d ** 2 + e ** 2 + f ** 2 + Delta ** 2 - 2 * a * mu + mu ** 2),
        -4 * b,
        1
    ])
    return array

#@jit
def GetAnalyticEnergies(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D):
    """
    H = A K[3, 0] + B K[0, 0] - mu K[3, 0] + C K[0, 2] + 
            D K[0, 1] + Delta K[1, 0] + E K[3, 2] + F K[3, 1]
    """
    energies = np.zeros(4)
    A = -2*w_0 * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
    B = 2*w_0 * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
    C = -B_y + Lambda_R * np.cos(k_x)*np.sin(phi_x) + Lambda_D * np.cos(k_y)*np.sin(phi_y)
    D = -B_x - Lambda_R * np.cos(k_y)*np.sin(phi_y) - Lambda_D * np.cos(k_x)*np.sin(phi_x)
    E = Lambda_R * np.sin(k_x)*np.cos(phi_x) + Lambda_D * np.sin(k_y)*np.cos(phi_y)
    F = -Lambda_R * np.sin(k_y)*np.cos(phi_y) - Lambda_D * np.sin(k_x)*np.cos(phi_x) 
    coefficient_array = get_coefficient_array(A, B, C, D, E, F, Delta, mu)      #1/2 because HBdG not considered in the coefficients
    c_0, c_1, c_2, c_3, c_4 = coefficient_array
    for m in range(4):
        energies[m] = 1/2 * np.real(get_cuartic_solution(c_0, c_1, c_2, c_3)[m])  #1/2 because HBdG not considered in the coefficients
    return energies

def GetSumOfPositiveAnalyticEnergy(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda_R, Lambda_D):
    """
    H = A K[3, 0] + B K[0, 0] - mu K[3, 0] + C K[0, 2] + 
            D K[0, 1] + Delta K[1, 0] + E K[3, 2] + F K[3, 1]
    """
    positive_energy = []
    A = -2*w_0 * (np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
    B = 2*w_0 * (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
    C = -B_y + Lambda_R * np.cos(k_x)*np.sin(phi_x) + Lambda_D * np.cos(k_y)*np.sin(phi_y)
    D = -B_x - Lambda_R * np.cos(k_y)*np.sin(phi_y) - Lambda_D * np.cos(k_x)*np.sin(phi_x)
    E = Lambda_R * np.sin(k_x)*np.cos(phi_x) + Lambda_D * np.sin(k_y)*np.cos(phi_y)
    F = -Lambda_R * np.sin(k_y)*np.cos(phi_y) - Lambda_D * np.sin(k_x)*np.cos(phi_x) 
    coefficient_array = get_coefficient_array(A, B, C, D, E, F, Delta, mu)      #1/2 because HBdG not considered in the coefficients
    c_0, c_1, c_2, c_3, c_4 = coefficient_array
    for m in range(4):
        energy = 1/2 * np.real(get_cuartic_solution(c_0, c_1, c_2, c_3)[m])  #1/2 because HBdG not considered in the coefficients
        if energy>0:
            positive_energy.append(energy)
    return np.sum(np.array(positive_energy))