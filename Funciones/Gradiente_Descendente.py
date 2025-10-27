def Gradiente_Descendente(variables: tuple, function, alfa: float, N_max: int, tolerance: float, x_0: list):
    """
    Implementación general del método de Gradiente Descendente para optimización.

    Parámetros:
        variables : tuple      → variables simbólicas (x,), (x,y), (x,y,z), ...
        function  : sympy.Expr → función simbólica f(x,...)
        alfa      : float      → tamaño de paso (learning rate)
        N_max     : int        → número máximo de iteraciones
        tolerance : float      → tolerancia (criterio de parada según ||∇f(x_k)||)
        x_0       : list       → punto inicial como lista o vector

    Retorna:
        x_k       : np.ndarray → punto aproximado del mínimo local
        trayectoria : list     → lista con los puntos intermedios (para graficar)
    """

    # Inicialización
    x_k = np.array(x_0, dtype=float)
    grad_f = [sp.diff(function, v) for v in variables]
    grad_func = sp.lambdify(variables, grad_f, "numpy")

    trayectoria = [x_k.copy()]  # Guardamos la trayectoria

    iteration = 0
    while iteration < N_max:
        # Calcular gradiente en el punto actual
        g_k = np.array(grad_func(*x_k), dtype=float).flatten()

        # Criterio de parada
        if np.linalg.norm(g_k) < tolerance:
            break

        # Actualización: x_{k+1} = x_k - alfa * gradiente
        x_k = x_k - alfa * g_k
        trayectoria.append(x_k.copy())
        iteration += 1

    return x_k, np.array(trayectoria)