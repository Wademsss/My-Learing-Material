# PHY 110L Homework 4
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Copilot was used as a code completion tool.

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Grid setup
# ============================================================
def make_grid(R=51):
    coords = np.mgrid[-R:R+1, -R:R+1]
    ygrid = coords[0]
    xgrid = coords[1]
    return xgrid, ygrid


# ============================================================
# Relaxation solver
# ============================================================
def relaxation_step(V, fixed_mask, periodic_x=False):
    Vnew = np.copy(V)

    if periodic_x:
        # top and bottom are normal fixed boundaries
        # left and right are periodic
        Vnew[1:-1, :] = (
            V[2:, :]
            + V[:-2, :]
            + np.roll(V, 1, axis=1)[1:-1, :]
            + np.roll(V, -1, axis=1)[1:-1, :]
        ) / 4
    else:
        Vnew[1:-1, 1:-1] = (
            V[2:, 1:-1]
            + V[:-2, 1:-1]
            + V[1:-1, 2:]
            + V[1:-1, :-2]
        ) / 4

    Vnew[fixed_mask] = V[fixed_mask]
    return Vnew


def solve_laplace(V, fixed_mask, tolerance=1e-3, max_iterations=200000, periodic_x=False):
    iterations = 0

    while iterations < max_iterations:
        Vnew = relaxation_step(V, fixed_mask, periodic_x=periodic_x)

        change = np.abs(Vnew - V)
        max_change = np.nanmax(change)

        iterations += 1

        if max_change < tolerance:
            return Vnew, iterations

        V = Vnew

    print("Warning: solution did not converge before max_iterations.")
    return V, iterations


# ============================================================
# Boundary helpers
# ============================================================
def add_outer_boundary(fixed_mask):
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True


def initialize_hollow_square(R=25):
    V = np.zeros((2*R + 1, 2*R + 1), dtype=float)
    fixed_mask = np.zeros_like(V, dtype=bool)

    add_outer_boundary(fixed_mask)

    center = R
    box_half = R // 5

    top = center + box_half
    bottom = center - box_half
    left = center - box_half
    right = center + box_half

    V[top, left:right+1] = 1
    V[bottom, left:right+1] = 1
    V[bottom:top+1, left] = 1
    V[bottom:top+1, right] = 1

    fixed_mask[top, left:right+1] = True
    fixed_mask[bottom, left:right+1] = True
    fixed_mask[bottom:top+1, left] = True
    fixed_mask[bottom:top+1, right] = True

    return V, fixed_mask


def initialize_creative_boundary(R=25):
    V = np.zeros((2*R + 1, 2*R + 1), dtype=float)
    fixed_mask = np.zeros_like(V, dtype=bool)

    add_outer_boundary(fixed_mask)

    center = R

    V[center-8:center+9, center-10:center-6] = 1.0
    fixed_mask[center-8:center+9, center-10:center-6] = True

    V[center-8:center+9, center+6:center+10] = -1.0
    fixed_mask[center-8:center+9, center+6:center+10] = True

    return V, fixed_mask


def initialize_parallel_plates(R=25):
    V = np.zeros((2*R + 1, 2*R + 1), dtype=float)
    fixed_mask = np.zeros_like(V, dtype=bool)

    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True

    center = R

    y_plus = center + int(0.2 * R)
    y_minus = center - int(0.2 * R)

    V[y_plus, :] = 1.0
    V[y_minus, :] = -1.0

    fixed_mask[y_plus, :] = True
    fixed_mask[y_minus, :] = True

    return V, fixed_mask


# ============================================================
# Plot helpers
# ============================================================
def plot_start_finish_cut(Vstart, Vfinal, R, title):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    extent = [-R, R, -R, R]

    im0 = ax[0, 0].imshow(Vstart, origin='lower', extent=extent, vmin=np.nanmin(Vfinal), vmax=np.nanmax(Vfinal))
    ax[0, 0].set_title("Start")
    ax[0, 0].set_aspect('equal')
    fig.colorbar(im0, ax=ax[0, 0])

    im1 = ax[1, 0].imshow(Vfinal, origin='lower', extent=extent, vmin=np.nanmin(Vfinal), vmax=np.nanmax(Vfinal))
    ax[1, 0].set_title("Finish")
    ax[1, 0].set_aspect('equal')
    fig.colorbar(im1, ax=ax[1, 0])

    center = R
    y_values = np.arange(-R, R+1)
    central_cut = Vfinal[:, center]

    ax[0, 1].plot(y_values, central_cut)
    ax[0, 1].set_title("Central Cut")
    ax[0, 1].set_xlabel("y")
    ax[0, 1].set_ylabel("V")

    ax[1, 1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()


def plot_single_result(Vfinal, R, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(
        Vfinal,
        origin='lower',
        extent=[-R, R, -R, R],
    )
    plt.colorbar(label='V')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()


# ============================================================
# Homework method
# ============================================================
def run_hw1():
    R = 25

    Vstart, fixed_mask = initialize_hollow_square(R)
    Vfinal, iterations = solve_laplace(
        np.copy(Vstart),
        fixed_mask,
        tolerance=1e-3
    )

    print("HW1 iterations:", iterations)

    plot_start_finish_cut(
        Vstart,
        Vfinal,
        R,
        title="HW1: Hollow Square Boundary"
    )


def run_hw2():
    print()
    print("HW2 computational cost:")

    for R in [25, 50, 100]:
        Vstart, fixed_mask = initialize_hollow_square(R)
        Vfinal, iterations = solve_laplace(
            np.copy(Vstart),
            fixed_mask,
            tolerance=1e-3
        )

        pixels = (2*R + 1) ** 2
        work_estimate = pixels * iterations

        print(
            "R =", R,
            "grid =", str(2*R + 1) + "x" + str(2*R + 1),
            "pixels =", pixels,
            "iterations =", iterations,
            "pixel-iterations =", work_estimate
        )


def run_hw3():
    R = 25

    Vstart, fixed_mask = initialize_creative_boundary(R)
    Vfinal, iterations = solve_laplace(
        np.copy(Vstart),
        fixed_mask,
        tolerance=1e-3
    )

    print("HW3 iterations:", iterations)

    plot_start_finish_cut(
        Vstart,
        Vfinal,
        R,
        title="HW3: Creative Boundary Variation"
    )


def run_hw4():
    R = 50

    Vstart, fixed_mask = initialize_parallel_plates(R)
    Vfinal, iterations = solve_laplace(
        np.copy(Vstart),
        fixed_mask,
        tolerance=1e-3,
        periodic_x=True
    )

    print("HW4 iterations:", iterations)

    plot_start_finish_cut(
        Vstart,
        Vfinal,
        R,
        title="HW4: Parallel Plates with Periodic x Boundary"
    )


# ============================================================
# Main
# ============================================================
def main():
    run_hw1()
    run_hw2()
    run_hw3()
    run_hw4()

    print()
    print("HW2 explanation:")
    print("As R increases, the number of pixels grows approximately like R^2.")
    print("The number of iterations also increases because information diffuses across a larger grid.")
    print("So the total computation grows faster than R^2.")

    print()
    print("HW4 explanation:")
    print("The solution does not become exactly V = 0 outside the plates because the plates are finite in y,")
    print("and the top and bottom boundaries are only a finite distance away.")
    print("Periodic left and right boundaries simulate infinite plates in x, but the finite grid still affects the result.")

    plt.show()


if __name__ == "__main__":
    main()