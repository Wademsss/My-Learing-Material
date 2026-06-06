# PHY 110L Homework 5
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Copilot was used as a code completion tool.

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# series solution
# ============================================================
def potential_series(x, y, L=1.0, V0=1.0, N=51):
    # V(x,y) sum over odd n

    V = np.zeros_like(np.asarray(x, dtype=float))

    for n in range(1, N + 1, 2):
        k = n * np.pi / L
        a_n = 4 * V0 / (n * np.pi)

        V += a_n * np.sin(k * x) * np.exp(-k * y)

    return V


def exact_potential(x, y, L=1.0, V0=1.0):

    numerator = np.sin(np.pi * x / L)
    denominator = np.sinh(np.pi * y / L)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = numerator / denominator
        V = (2 * V0 / np.pi) * np.arctan(ratio)

    # At y = 0 and 0 < x < L, the exact boundary condition is V = V0.
    V = np.where((y == 0) & (x > 0) & (x < L), V0, V)

    return V


# ============================================================
# Electric field 
# ============================================================
def electric_field_series(x, y, L=1.0, V0=1.0, N=51):
    
    # E = -grad(V)

    Ex = 0.0
    Ey = 0.0

    for n in range(1, N + 1, 2):
        k = n * np.pi / L
        a_n = 4 * V0 / (n * np.pi)

        Ex += -a_n * k * np.cos(k * x) * np.exp(-k * y)
        Ey += a_n * k * np.sin(k * x) * np.exp(-k * y)

    return Ex, Ey


def unit_field(x, y, L=1.0, V0=1.0, N=51):
    Ex, Ey = electric_field_series(x, y, L=L, V0=V0, N=N)
    mag = np.hypot(Ex, Ey)

    if mag == 0 or not np.isfinite(mag):
        return None

    return Ex / mag, Ey / mag


# ============================================================
# field-line tracing
# ============================================================
def rk2_field_step(x, y, h, L=1.0, V0=1.0, N=51):
    first = unit_field(x, y, L=L, V0=V0, N=N)

    if first is None:
        return None

    ux1, uy1 = first

    xm = x + 0.5 * h * ux1
    ym = y + 0.5 * h * uy1

    second = unit_field(xm, ym, L=L, V0=V0, N=N)

    if second is None:
        return None

    ux2, uy2 = second

    return x + h * ux2, y + h * uy2


def trace_field_line(
    ax,
    x0,
    y0,
    L=1.0,
    V0=1.0,
    N=51,
    h=0.005,
    max_steps=5000,
    color='black',
    linewidth=0.8
):
    x = float(x0)
    y = float(y0)

    xs = [x]
    ys = [y]

    for i in range(max_steps):
        next_point = rk2_field_step(x, y, h, L=L, V0=V0, N=N)

        if next_point is None:
            break

        x_new, y_new = next_point

        xs.append(x_new)
        ys.append(y_new)

        x, y = x_new, y_new

        # Termination conditions
        if x <= 0 or x >= L:
            break

        if y >= 2 * L:
            break

        if y < 0:
            break

    ax.plot(xs, ys, color=color, linewidth=linewidth)


# ============================================================
# Grid helper
# ============================================================
def make_grid(L=1.0, nx=151, ny=301):
    x = np.linspace(0, L, nx)
    y = np.linspace(0, 2 * L, ny)

    xgrid, ygrid = np.meshgrid(x, y)

    return xgrid, ygrid


# ============================================================
# Plot helpers
# ============================================================
def plot_3d_potential(xgrid, ygrid, V, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xgrid, ygrid, V, linewidth=0, antialiased=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V')
    ax.set_title(title)

    fig.tight_layout()


def plot_boundary_match(xgrid, V, L=1.0, V0=1.0, title='Boundary at y = 0'):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = xgrid[0, :]
    V_bottom = V[0, :]

    ax.plot(x, V_bottom, label='Series approximation')
    ax.axhline(V0, linestyle='--', label='Target V0')

    ax.set_xlabel('x')
    ax.set_ylabel('V(x,0)')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()


def plot_2d_potential_with_field_lines(xgrid, ygrid, V, L=1.0, V0=1.0, N=51):
    fig, ax = plt.subplots(figsize=(7, 8))

    im = ax.imshow(
        V,
        origin='lower',
        extent=[0, L, 0, 2 * L],
        aspect='auto'
    )

    fig.colorbar(im, ax=ax, label='V')

    # Launch 50 uniformly spaced field lines from the y = 0 boundary.
    xs = np.linspace(0.02 * L, 0.98 * L, 50)

    for x0 in xs:
        trace_field_line(
            ax,
            x0=x0,
            y0=0.001 * L,
            L=L,
            V0=V0,
            N=N,
            h=0.004 * L,
            color='black',
            linewidth=0.7
        )

    ax.set_xlim(0, L)
    ax.set_ylim(0, 2 * L)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('HW2: Field Lines from y = 0 Boundary')

    fig.tight_layout()


def plot_fractional_error(error, L=1.0):
    fig, ax = plt.subplots(figsize=(7, 8))

    # some where wrong idk...
    im = ax.imshow(
        error,
        origin='lower',
        extent=[0, L, 0, 2 * L],
        aspect='auto',
        cmap='coolwarm',
        vmin=0.0009,
        vmax=0.001
    )

    fig.colorbar(im, ax=ax, label='Fractional error')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('HW3: Fractional Error of Series Approximation')

    fig.tight_layout()


# ============================================================
# Error helper
# ============================================================
def fractional_error(V_approx, V_exact):
    error = np.full_like(V_exact, np.nan, dtype=float)

    mask = np.isfinite(V_approx) & np.isfinite(V_exact) & (np.abs(V_exact) > 1e-8)

    error[mask] = (V_approx[mask] - V_exact[mask]) / V_exact[mask]

    return error


# ============================================================
# tasks
# ============================================================
def run_hw1(L, V0, N):
    xgrid, ygrid = make_grid(L=L)

    V = potential_series(
        xgrid,
        ygrid,
        L=L,
        V0=V0,
        N=N
    )

    plot_3d_potential(
        xgrid,
        ygrid,
        V,
        title=f'HW1: 3D Series Solution, N = {N}'
    )

    plot_boundary_match(
        xgrid,
        V,
        L=L,
        V0=V0,
        title=f'HW1: Boundary Match at y = 0, N = {N}'
    )


def run_hw2(L, V0, N):
    xgrid, ygrid = make_grid(L=L)

    V = potential_series(
        xgrid,
        ygrid,
        L=L,
        V0=V0,
        N=N
    )

    plot_2d_potential_with_field_lines(
        xgrid,
        ygrid,
        V,
        L=L,
        V0=V0,
        N=N
    )


def run_hw3(L, V0, N):
    xgrid, ygrid = make_grid(L=L)
    

    V_approx = potential_series(
        xgrid,
        ygrid,
        L=L,
        V0=V0,
        N=N
    )

    V_exact = exact_potential(
        xgrid,
        ygrid,
        L=L,
        V0=V0
    )

    error = fractional_error(V_approx, V_exact)

    plot_fractional_error(
        error,
        L=L
    )


# ============================================================
# Main
# ============================================================
def main():
    
    L = 1.0
    V0 = 1.0

    N = 51

    run_hw1(L, V0, N)
    run_hw2(L, V0, N)
    run_hw3(L, V0, N)

    print("HW1:")
    print("As N increases, the bottom boundary y = 0 better matches V = V0.The largest mismatch is near x = 0 and x = L because the boundary condition jumps there.")
    print()
   
    print("HW2:")
    print("The field lines are perpendicular to the equipotentials because E = -grad(V). Near the bottom boundary, many field lines rise upward from the high-potential plate.Near the side walls, field lines bend toward the grounded side boundaries.")
    print()

    print("HW3:")
    print("The largest fractional errors occur near the bottom corners where the boundary condition is discontinuous.The finite Fourier series struggles near discontinuities, producing Gibbs-like behavior.Away from the y = 0 boundary, higher-n terms decay rapidly, so the approximation improves.")

    plt.show()


if __name__ == "__main__":
    main()