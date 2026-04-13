import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Charge class
# =========================================================
class Charge:
    def __init__(self, x, y, q):
        self.x = x
        self.y = y
        self.q = q

    def potential(self, xgrid, ygrid):
        dx = xgrid - self.x
        dy = ygrid - self.y
        r = np.hypot(dx, dy)

        # avoid division by zero at the charge location
        with np.errstate(divide='ignore', invalid='ignore'):
            V = self.q / r
        return V


# =========================================================
# Grid setup
# =========================================================
def make_grid(n=51, xmin=-25, xmax=25, ymin=-25, ymax=25):
    coords = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    ygrid = coords[0, :, :]
    xgrid = coords[1, :, :]
    return xgrid, ygrid


# =========================================================
# Physics helpers
# =========================================================
def total_potential(charges, xgrid, ygrid):
    V = np.zeros_like(xgrid, dtype=float)
    for charge in charges:
        V += charge.potential(xgrid, ygrid)
    return V


def finite_contour_levels(V, num_levels=10):
    finite_vals = V[np.isfinite(V)]
    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)

    # If both positive and negative values exist, make symmetric levels
    if vmin < 0 and vmax > 0:
        vmax_abs = max(abs(vmin), abs(vmax))
        levels = np.linspace(-vmax_abs, vmax_abs, num_levels)
    else:
        levels = np.linspace(vmin, vmax, num_levels)

    return levels


def electric_field_from_potential(V):
    # np.gradient returns [dV/dy, dV/dx]
    dV_dy, dV_dx = np.gradient(V)
    Ex = -dV_dx
    Ey = -dV_dy
    return Ex, Ey


def field_direction_degrees(Ex, Ey):
    theta = np.degrees(np.arctan2(Ey, Ex))
    theta = (theta + 360) % 360
    return theta


# =========================================================
# Plot helpers
# =========================================================
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


def plot_field_direction_from_potential(charges, xgrid, ygrid, title):
    V = total_potential(charges, xgrid, ygrid)
    Ex, Ey = electric_field_from_potential(V)
    theta = field_direction_degrees(Ex, Ey)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        theta,
        origin='lower',
        extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
        cmap='hsv',
        vmin=0,
        vmax=360
    )
    plt.colorbar(label='Field direction (degrees)')

    for c in charges:
        marker = 'o' if c.q > 0 else 'x'
        plt.scatter(c.x, c.y, marker=marker, s=100)

    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()


# =========================================================
# Homework 2: check dipole far-field behavior
# =========================================================
def dipole_far_field_check():
    # Dipole along x-axis: +1 at (-3.5,0), -1 at (3.5,0)
    # Along y-axis (x = 0), for large |y|:
    # V ~ p*y / r^3, and since x=0 here, |V| ~ const / y^2
    y = np.array([10, 12, 15, 18, 20, 22, 25], dtype=float)

    q = 1.0
    a = 3.5

    r_plus = np.hypot(0 - (-a), y - 0)
    r_minus = np.hypot(0 - a, y - 0)

    V = q / r_plus - q / r_minus

    # compare |V| * y^2 ; if nearly constant, then V ~ 1/y^2
    check = np.abs(V) * y**2

    print("\nDipole far-field check along x=0:")
    print(" y         V(y)              |V|*y^2")
    for yi, Vi, Ci in zip(y, V, check):
        print(f"{yi:5.1f}   {Vi: .8e}   {Ci: .8e}")


# =========================================================
# Main
# =========================================================
def main():
    xgrid, ygrid = make_grid()

    # -----------------------------------------------------
    # HW1: two negative charges separated by 14 pixels
    # -----------------------------------------------------
    charges1 = [
        Charge(-7, 0, -1),
        Charge(7, 0, -1)
    ]
    plot_potential(charges1, xgrid, ygrid, 'HW1: Two Negative Charges')

    # -----------------------------------------------------
    # HW2: dipole
    # -----------------------------------------------------
    charges2 = [
        Charge(-3.5, 0, +1),
        Charge(3.5, 0, -1)
    ]
    plot_potential(charges2, xgrid, ygrid, 'HW2: Dipole Potential')
    dipole_far_field_check()

    # -----------------------------------------------------
    # HW3: more complicated arrangement
    # Example: two dipoles
    # -----------------------------------------------------
    charges3 = [
        Charge(-10, 0, +1),
        Charge(-4, 0, -1),
        Charge(4, 0, +1),
        Charge(10, 0, -1)
    ]
    plot_potential(charges3, xgrid, ygrid, 'HW3: Two Dipoles Arrangement')

    # -----------------------------------------------------
    # HW4: E field direction from dipole potential
    # -----------------------------------------------------
    plot_field_direction_from_potential(
        charges2, xgrid, ygrid,
        'HW4: Electric Field Direction from Dipole Potential'
    )

    plt.show()


if __name__ == "__main__":
    main()