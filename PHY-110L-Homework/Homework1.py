import numpy as np
import matplotlib.pyplot as plt

# Grid setup
def make_grid(n=51, xmin=-25, xmax=25, ymin=-25, ymax=25):
    coords = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    ygrid = coords[0, :, :]
    xgrid = coords[1, :, :]
    return xgrid, ygrid

# Charge class
class Charge:
    def __init__(self, x, y, q):
        self.x = x
        self.y = y
        self.q = q

    def potential(self, xgrid, ygrid):
        r = np.hypot(xgrid - self.x, ygrid - self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            V = self.q / r
        return V
    
# Physics helpers
def total_potential(charges, xgrid, ygrid):
    Vtotal = np.zeros_like(xgrid, dtype=float)
    for c in charges:
        Vtotal += c.potential(xgrid, ygrid)
    return Vtotal

def finite_contour_levels(V, num_levels=10):
    finite_vals = V[np.isfinite(V)]
    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)

    if vmin < 0 and vmax > 0:
        vmax_abs = max(abs(vmin), abs(vmax))
        levels = np.linspace(-vmax_abs, vmax_abs, num_levels)
    else:
        levels = np.linspace(vmin, vmax, num_levels)

    return levels

# Plot helpers
def plot_potential(charges, xgrid, ygrid, title):
        V = total_potential(charges, xgrid, ygrid)

        plt.figure(figsize=(7, 6))
        plt.imshow(
            V,
            origin='lower',
            extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()]
        )
        plt.colorbar(label='Potential V')

        levels = finite_contour_levels(V, num_levels=10)
        plt.contour(
            xgrid, ygrid, V,
            levels=levels,
            colors='black',
            linewidths=0.8
        )

        for c in charges:
            marker = 'o' if c.q > 0 else 'x'
            plt.scatter(c.x, c.y, marker=marker, s=100)

        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.tight_layout()
        return V

# Main
def main():
    xgrid, ygrid = make_grid()

    # HW1: Two charges
    charges1 = [Charge(-7, 0, -1), Charge(7, 0, -1)]
    plot_potential(charges1, xgrid, ygrid, title='HW1: Two Charges')

    # HW2: Dipole
    charges2 = [Charge(-3.5, 0, +1), Charge(3.5, 0, -1)]
    plot_potential(charges2, xgrid, ygrid, title='HW2: Dipole')

    plt.show()

if __name__ == "__main__":
    main()