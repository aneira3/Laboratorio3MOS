import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Newton_Raphson(variables: tuple, function, alfa: float, N_max: int, tolerance: float, x_0: list):
    """
    Implementación general del método de Newton-Raphson para optimización.
    - Funciona para 1D y nD (n >= 1)
    
    Parámetros:
        variables : tuple      → variables simbólicas (x,), (x,y), (x,y,z), ...
        function  : sympy.Expr → función simbólica f(x,...)
        alfa      : float      → factor de paso
        N_max     : int        → número máximo de iteraciones
        tolerance : float      → tolerancia (norma del gradiente)
        x_0       : list       → lista o vector con el punto inicial
    
    Retorna:
        x_k : np.ndarray       → punto crítico aproximado
        H_eval : sympy.Matrix  → Hessiana evaluada en x_k (para criterio)
    """

    n = len(variables)
    x_k = np.array(x_0, dtype=float)  # Siempre vector, incluso en 1D

    # Caso general: gradiente y Hessiana simbólicos
    grad_f = [sp.diff(function, v) for v in variables]
    H_f = sp.hessian(function, variables)

    # Funciones evaluables numéricamente
    grad_func = sp.lambdify(variables, grad_f, "numpy")
    H_func = sp.lambdify(variables, H_f, "numpy")

    iteration = 0

    while iteration < N_max:
        # Evaluar gradiente y Hessiana en el punto actual
        g = np.array(grad_func(*x_k), dtype=float).flatten()
        H = np.array(H_func(*x_k), dtype=float)

        # Criterio de parada: norma del gradiente
        if np.linalg.norm(g) < tolerance:
            break

        # Verificar invertibilidad
        if np.linalg.det(H) == 0:
            print("Hessiana no se puede invertir en la mátriz:\n")
            print(H)
            break

        # Actualizar x_k
        d_k = -np.linalg.solve(H, g)
        x_k = x_k + alfa * d_k
        iteration += 1

    # Evaluar Hessiana en el punto final (para criterio)
    H_eval = H_f.subs({v: x_k[i] for i, v in enumerate(variables)})

    return x_k, H_eval


def Criterio_Segunda_Derivada(H_eval):
    """
    Clasifica el punto crítico según los eigenvalues de la Hessiana.
    Compatible con 1D y nD.
    """
    eigenvals = [float(val) for val in H_eval.eigenvals().keys()]

    if all(val > 0 for val in eigenvals):
        return "Mínimo local (H definida positiva)"
    elif all(val < 0 for val in eigenvals):
        return "Máximo local (H definida negativa)"
    else:
        return "Punto de silla (H indefinida)"
