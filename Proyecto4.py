import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from EDsNumericasFunciones import Euler, Trapecio, PuntoMedio, RK4


def masa_resorte(t, X):
    x1, x1_dot, x2, x2_dot, x3, x3_dot = X
    
    m1, m2, m3 = 1.0, 1.0, 1.0
    k1, k2, k3 = 1.0, 1.0, 1.0
    
    x1_ddot = (-k1 * x1 + k2 * (x2 - x1)) / m1
    x2_ddot = (-k2 * (x2 - x1) + k3 * (x3 - x2)) / m2
    x3_ddot = -k3 * (x3 - x2) / m3
    
    return [x1_dot, x1_ddot, x2_dot, x2_ddot, x3_dot, x3_ddot]

def solve_system(method, F, y0, t0, tN, h=0.01):
    if method == 'Euler':
        return Euler(F, y0, t0, tN, h)
    elif method == 'Trapecio':
        return Trapecio(F, y0, t0, tN, h)
    elif method == 'Punto Medio':
        return PuntoMedio(F, y0, t0, tN, h)
    elif method == 'RK4':
        return RK4(F, y0, t0, tN, h)
    else:
        raise ValueError("Método no válido")
    
    
# Ejemplo de uso
t0, tN = 0, 10
h = 0.01
y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Método de Euler
t_euler, y_euler = solve_system('Euler', masa_resorte, y0, t0, tN, h)

# Método del Trapecio
t_trap, y_trap = solve_system('Trapecio', masa_resorte, y0, t0, tN, h)

# Método del Punto Medio
t_pm, y_pm = solve_system('Punto Medio', masa_resorte, y0, t0, tN, h)

# Método RK4
t_rk4, y_rk4 = solve_system('RK4', masa_resorte, y0, t0, tN, h)

# Solución analítica
sol = solve_ivp(masa_resorte, [t0, tN], y0, t_eval=np.linspace(t0, tN, int((tN-t0)/h)))

# Gráficos
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t_euler, y_euler)
plt.title('Euler Method')

plt.subplot(2, 2, 2)
plt.plot(t_trap, y_trap)
plt.title('Trapezoidal Method')

plt.subplot(2, 2, 3)
plt.plot(t_pm, y_pm)
plt.title('Midpoint Method')

plt.subplot(2, 2, 4)
plt.plot(t_rk4, y_rk4)
plt.title('RK4 Method')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], label='Analytical Solution', linewidth=2)
plt.plot(t_euler, y_euler[:, 0], label='Euler Method', linestyle='--')
plt.plot(t_trap, y_trap[:, 0], label='Trapezoidal Method', linestyle='--')
plt.plot(t_pm, y_pm[:, 0], label='Midpoint Method', linestyle='--')
plt.plot(t_rk4, y_rk4[:, 0], label='RK4 Method', linestyle='--')

plt.title('Comparative Analysis - Mass 1')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.grid(True)
plt.show()