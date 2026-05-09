import numpy as np
import matplotlib.pyplot as plt

alpha = 0.003

def vg_over_c(x, Gamma):
    D = (1 - x)**2 + Gamma**2 * x
    
    n = 1 + alpha * (1 - x) / D
    
    dDdx = -2 * (1 - x) + Gamma**2
    
    dndx = alpha * (-D - (1 - x) * dDdx) / D**2
    
    return 1 / (n + 2 * x * dndx)

x = np.linspace(0, 2, 5000)

# Case (a): gamma = 0
Gamma_a = 0

# Avoid exact resonance x = 1
x_a = x[np.abs(x - 1) > 1e-3]
y_a = vg_over_c(x_a, Gamma_a)

# Case (b): gamma = 0.1 omega_0
Gamma_b = 0.1
y_b = vg_over_c(x, Gamma_b)

plt.figure(figsize=(8, 5))
plt.plot(x_a, y_a, label=r"$\gamma=0$")
plt.plot(x, y_b, label=r"$\gamma=0.1\omega_0$")
plt.axhline(1, linestyle="--", label=r"$v_g=c$")
plt.xlabel(r"$x=(\omega/\omega_0)^2$")
plt.ylabel(r"$v_g/c$")
plt.ylim(-5, 5)
plt.legend()
plt.grid(True)
plt.show()