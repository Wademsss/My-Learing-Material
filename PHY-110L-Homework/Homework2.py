# PHY 110L Homework 1
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Colpiolt was used as a code completion tool.
import numpy as np
import matplotlib.pyplot as plt



# 1. Grid

def make_grid(xmin=-25, xmax=25, ymin=-25, ymax=25):
    coords = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
    ygrid = coords[0]
    xgrid = coords[1]
    return xgrid, ygrid



# 2. Charge class

class Charge:
    def __init__(self, x, y, q):
        self.x = float(x)
        self.y = float(y)
        self.q = float(q)

    def potential(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = np.hypot(dx, dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            V = self.q / r
        return V

    def field(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = np.hypot(dx, dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            Ex = self.q * dx / (r ** 3)
            Ey = self.q * dy / (r ** 3)
        return Ex, Ey



# 3. Superposition helpers

def total_potential(charges, xgrid, ygrid):
    V = np.zeros_like(xgrid, dtype=float)
    for c in charges:
        V += c.potential(xgrid, ygrid)
    return V


def total_field_on_grid(charges, xgrid, ygrid):
    Ex = np.zeros_like(xgrid, dtype=float)
    Ey = np.zeros_like(ygrid, dtype=float)
    for c in charges:
        ex, ey = c.field(xgrid, ygrid)
        Ex += ex
        Ey += ey
    return Ex, Ey


def total_field_at_point(charges, x, y):
    Ex_total = 0.0
    Ey_total = 0.0
    for c in charges:
        ex, ey = c.field(x, y)
        Ex_total += ex
        Ey_total += ey
    return Ex_total, Ey_total



# 4. Problem 1: centered-difference field from potential
#    np.gradient(V) returns (dV/dy, dV/dx)
#    E = -grad(V), so return (Ey, Ex) for easy comparison

def centered_difference_field(V, dx=1.0, dy=1.0):
    Ey = np.full_like(V, np.nan, dtype=float)
    Ex = np.full_like(V, np.nan, dtype=float)

    # interior points only
    Ey[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2.0 * dy)
    Ex[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2.0 * dx)

    return Ey, Ex


def numpy_gradient_field(V, dx=1.0, dy=1.0):
    dVdy, dVdx = np.gradient(V, dy, dx)
    Ey = -dVdy
    Ex = -dVdx
    return Ey, Ex



# 5. Problem 2: fractional error

def fractional_error(approx, true, eps=1e-12):
    err = np.full_like(true, np.nan, dtype=float)
    mask = np.isfinite(approx) & np.isfinite(true) & (np.abs(true) > eps)
    err[mask] = (approx[mask] - true[mask]) / true[mask]
    return err



# 6. Problem 3/4: Euler field-line tracing

def trace_field_line(
    charges,
    x0,
    y0,
    terminate_at,
    direction="withfield",
    step=0.35,
    max_steps=4000,
    stop_radius=0.8,
    bounds=(-25, 25, -25, 25),
    color="k",
    linewidth=1.0,
):
    x = float(x0)
    y = float(y0)
    xmin, xmax, ymin, ymax = bounds

    for _ in range(max_steps):
        Ex, Ey = total_field_at_point(charges, x, y)

        if not np.isfinite(Ex) or not np.isfinite(Ey):
            break

        norm = np.hypot(Ex, Ey)
        if norm == 0 or not np.isfinite(norm):
            break

        ux = Ex / norm
        uy = Ey / norm

        if direction == "againstfield":
            ux = -ux
            uy = -uy

        x_new = x + step * ux
        y_new = y + step * uy

        plt.plot([x, x_new], [y, y_new], color=color, linewidth=linewidth)

        x, y = x_new, y_new

        # termination near target charge
        if np.hypot(x - terminate_at[0], y - terminate_at[1]) < stop_radius:
            break

        # stop if outside plot box
        if x < xmin or x > xmax or y < ymin or y > ymax:
            break


def launch_lines_from_charge(
    source_charge,
    target_charge,
    charges,
    n_lines=8,
    launch_radius=0.8,
    step=0.35,
    color="k",
    linewidth=1.0,
):
    # avoid Cartesian axes as suggested in the handout:
    # for n=8 -> 22.5°, 67.5°, 112.5°, ...
    angles = np.linspace(0, 2 * np.pi, n_lines, endpoint=False) + np.pi / n_lines

    for theta in angles:
        x0 = source_charge.x + launch_radius * np.cos(theta)
        y0 = source_charge.y + launch_radius * np.sin(theta)

        direction = "withfield" if source_charge.q > 0 else "againstfield"

        trace_field_line(
            charges=charges,
            x0=x0,
            y0=y0,
            terminate_at=(target_charge.x, target_charge.y),
            direction=direction,
            step=step,
            stop_radius=launch_radius,
            color=color,
            linewidth=linewidth,
        )



# 7. Plot helpers

def show_imshow_with_colorbar(data, title, extent=(-25, 25, -25, 25), cmap=None):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
    )
    plt.colorbar(im)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()



# 8. Main homework driver

def main():
    # Dipole setup used throughout
 
    xgrid, ygrid = make_grid()
    dx = 1.0
    dy = 1.0

    positive = Charge(-7, 0, +1)
    negative = Charge(+7, 0, -1)
    charges = [positive, negative]

    V = total_potential(charges, xgrid, ygrid)

    
    # Problem 1
    # Implement second-order centered difference and compare
    # with np.gradient on the same gridded dipole potential
    
    my_Ey, my_Ex = centered_difference_field(V, dx=dx, dy=dy)
    np_Ey, np_Ex = numpy_gradient_field(V, dx=dx, dy=dy)

    interior = np.s_[1:-1, 1:-1]
    mask_x = np.isfinite(my_Ex[interior]) & np.isfinite(np_Ex[interior])
    mask_y = np.isfinite(my_Ey[interior]) & np.isfinite(np_Ey[interior])

    max_diff_Ex = np.max(np.abs(my_Ex[interior][mask_x] - np_Ex[interior][mask_x]))
    max_diff_Ey = np.max(np.abs(my_Ey[interior][mask_y] - np_Ey[interior][mask_y]))

    print("Problem 1:")
    print("Max |my_Ex - np_Ex| on interior =", max_diff_Ex)
    print("Max |my_Ey - np_Ey| on interior =", max_diff_Ey)
    print()

    # Optional visual confirmation
    show_imshow_with_colorbar(my_Ex, "Problem 1: My centered-difference Ex")
    show_imshow_with_colorbar(np_Ex, "Problem 1: NumPy-gradient Ex")
    show_imshow_with_colorbar(my_Ey, "Problem 1: My centered-difference Ey")
    show_imshow_with_colorbar(np_Ey, "Problem 1: NumPy-gradient Ey")

    
    # Problem 2
    # True field from Coulomb law, then fractional error of
    # the gradient-based approximation
    
    true_Ex, true_Ey = total_field_on_grid(charges, xgrid, ygrid)

    err_Ex = fractional_error(my_Ex, true_Ex)
    err_Ey = fractional_error(my_Ey, true_Ey)

    print("Problem 2:")
    finite_x = np.isfinite(err_Ex)
    finite_y = np.isfinite(err_Ey)
    print("Max |fractional error Ex| =", np.nanmax(np.abs(err_Ex[finite_x])))
    print("Max |fractional error Ey| =", np.nanmax(np.abs(err_Ey[finite_y])))
    print("Machine precision for float64 is about 1e-16, so these errors are much larger.")
    print()

    show_imshow_with_colorbar(
        err_Ex,
        "Problem 2: Fractional error in Ex",
        cmap="coolwarm",
    )
    show_imshow_with_colorbar(
        err_Ey,
        "Problem 2: Fractional error in Ey",
        cmap="coolwarm",
    )

    
    # Problem 3(a)
    # Launch from positive charge straight toward negative
    
    plt.figure(figsize=(7, 6))
    trace_field_line(
        charges=charges,
        x0=positive.x + 0.8,
        y0=positive.y,
        terminate_at=(negative.x, negative.y),
        direction="withfield",
        step=0.35,
        stop_radius=0.8,
        color="k",
        linewidth=1.5,
    )
    plt.scatter(positive.x, positive.y, s=160, c="red")
    plt.scatter(negative.x, negative.y, s=160, c="blue")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 3(a): From + charge toward - charge")
    plt.tight_layout()

    
    # Problem 3(b)
    # Launch from negative charge straight toward positive
    # Must go against the field direction
    
    plt.figure(figsize=(7, 6))
    trace_field_line(
        charges=charges,
        x0=negative.x - 0.8,
        y0=negative.y,
        terminate_at=(positive.x, positive.y),
        direction="againstfield",
        step=0.35,
        stop_radius=0.8,
        color="k",
        linewidth=1.5,
    )
    plt.scatter(positive.x, positive.y, s=160, c="red")
    plt.scatter(negative.x, negative.y, s=160, c="blue")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 3(b): From - charge toward + charge")
    plt.tight_layout()

    
    # Problem 3(c)
    # Launch from positive charge perpendicular to dipole axis
    
    plt.figure(figsize=(7, 6))
    trace_field_line(
        charges=charges,
        x0=positive.x,
        y0=positive.y + 0.8,
        terminate_at=(negative.x, negative.y),
        direction="withfield",
        step=0.35,
        stop_radius=0.8,
        color="k",
        linewidth=1.5,
    )
    plt.scatter(positive.x, positive.y, s=160, c="red")
    plt.scatter(negative.x, negative.y, s=160, c="blue")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 3(c): From + charge perpendicular to dipole axis")
    plt.tight_layout()

    
    # Problem 4
    # Final attractive figure with at least 8 lines connected
    # to each charge
    
    plt.figure(figsize=(8, 8))

    launch_lines_from_charge(
        source_charge=positive,
        target_charge=negative,
        charges=charges,
        n_lines=8,
        launch_radius=0.8,
        step=0.30,
        color="k",
        linewidth=1.1,
    )

    launch_lines_from_charge(
        source_charge=negative,
        target_charge=positive,
        charges=charges,
        n_lines=8,
        launch_radius=0.8,
        step=0.30,
        color="k",
        linewidth=1.1,
    )

    plt.scatter(positive.x, positive.y, s=260, c="red", edgecolors="black", zorder=5)
    plt.scatter(negative.x, negative.y, s=260, c="blue", edgecolors="black", zorder=5)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 4: Dipole field lines")
    plt.tight_layout()

    plt.show()

    
    # Problem 3(d) written answer
    
    print("Problem 3(d) short answer:")
    print(
        "A field line is not the trajectory of a charged particle. "
        "A field line is defined so that its tangent is always parallel to E at each point. "
        "A particle trajectory is determined by Newton's second law, F = qE = ma, so it depends "
        "on the particle's mass, charge, and velocity history."
    )


if __name__ == "__main__":
    main()