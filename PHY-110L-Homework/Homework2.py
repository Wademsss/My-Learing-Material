# PHY 110L Homework 2
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Colpiolt was used as a code completion tool.
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

    def field(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = np.hypot(dx, dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            Ex = self.q * dx / (r ** 3)
            Ey = self.q * dy / (r ** 3)
        return Ex, Ey

# Physics helpers

# HW1 Part
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

# HW2 Part
def total_field(charges, x, y):
    Ex_total = 0
    Ey_total = 0

    for c in charges:
        Ex, Ey = c.field(x, y)
        Ex_total += Ex
        Ey_total += Ey

    return Ex_total, Ey_total

def electric_field_from_potential(V):
    # np.gradient returns [dV/dy, dV/dx]
    dV_dy, dV_dx = np.gradient(V)
    Ex = -dV_dx
    Ey = -dV_dy
    return Ex, Ey

def my_centered_difference_field(V, dx=1, dy=1):
    Ex = np.full_like(V, np.nan, dtype=float)
    Ey = np.full_like(V, np.nan, dtype=float)

    # centered difference, interior points only
    Ex[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2 * dx)
    Ey[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)

    return Ex, Ey

def fractional_error(approx, true):
    error = np.full_like(true, np.nan, dtype=float)

    mask = (
        np.isfinite(approx)
        & np.isfinite(true)
        & (np.abs(true) > 1e-12)
    )

    error[mask] = (approx[mask] - true[mask]) / true[mask]

    return error

# Field line helpers
def trace_field_line(
    charges,
    x0,
    y0,
    stop_x,
    stop_y,
    step=0.35,
    stop_radius=0.8,
    max_steps=3000,
    direction='withfield'
):
    x = x0
    y = y0

    for i in range(max_steps):
        Ex, Ey = total_field(charges, x, y)

        if not np.isfinite(Ex) or not np.isfinite(Ey):
            break

        E_mag = np.hypot(Ex, Ey)

        if E_mag == 0:
            break

        ux = Ex / E_mag
        uy = Ey / E_mag

        if direction == 'againstfield':
            ux = -ux
            uy = -uy

        x_new = x + step * ux
        y_new = y + step * uy

        plt.plot([x, x_new], [y, y_new], color='black', linewidth=1)

        x = x_new
        y = y_new

        if np.hypot(x - stop_x, y - stop_y) < stop_radius:
            break

        if x < -25 or x > 25 or y < -25 or y > 25:
            break


def launch_lines_from_charge(
    charges,
    source_charge,
    target_charge,
    number_of_lines=8,
    launch_radius=0.8
):
    angles = np.linspace(0, 2*np.pi, number_of_lines, endpoint=False)

    # offset angle to avoid launching exactly along axes
    angles = angles + np.pi / number_of_lines

    for theta in angles:
        x0 = source_charge.x + launch_radius * np.cos(theta)
        y0 = source_charge.y + launch_radius * np.sin(theta)

        if source_charge.q > 0:
            direction = 'withfield'
        else:
            direction = 'againstfield'

        trace_field_line(
            charges,
            x0,
            y0,
            target_charge.x,
            target_charge.y,
            direction=direction
        )

# Plot helpers
def plot_fractional_error(error, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(
        error,
        origin='lower',
        extent=[-25, 25, -25, 25],
        cmap='coolwarm'
    )
    plt.colorbar(label='Fractional error')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()


def plot_field_lines(
    charges,
    positive,
    negative,
    start_points,
    title
):
    plt.figure(figsize=(7, 6))

    for start in start_points:
        start_x = start[0]
        start_y = start[1]
        direction = start[2]
        stop_charge = start[3]

        trace_field_line(
            charges,
            start_x,
            start_y,
            stop_charge.x,
            stop_charge.y,
            direction=direction
        )

    plt.scatter(
        positive.x,
        positive.y,
        color='red',
        s=220,
        edgecolors='black',
        zorder=5
    )

    plt.scatter(
        negative.x,
        negative.y,
        color='blue',
        s=220,
        edgecolors='black',
        zorder=5
    )

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()

def make_start_points_from_charge(
    source_charge,
    target_charge,
    number_of_lines=16,
    launch_radius=0.8
):
    start_points = []

    angles = np.linspace(0, 2*np.pi, number_of_lines, endpoint=False)
    angles = angles + np.pi / number_of_lines

    for theta in angles:
        x0 = source_charge.x + launch_radius * np.cos(theta)
        y0 = source_charge.y + launch_radius * np.sin(theta)

        if source_charge.q > 0:
            direction = 'withfield'
        else:
            direction = 'againstfield'

        start_points.append((x0, y0, direction, target_charge))

    return start_points

# Main
def main():
    xgrid, ygrid = make_grid()

    positive = Charge(-7, 0, +1)
    negative = Charge(7, 0, -1)
    charges = [positive, negative]

    V = total_potential(charges, xgrid, ygrid)

    # HW1: Computing the gradient yourself
    my_Ex, my_Ey = my_centered_difference_field(V)
    np_Ex, np_Ey = electric_field_from_potential(V)

    interior = np.s_[1:-1, 1:-1]

    diff_Ex = np.nanmax(np.abs(my_Ex[interior] - np_Ex[interior]))
    diff_Ey = np.nanmax(np.abs(my_Ey[interior] - np_Ey[interior]))

    print("HW1: Centered difference compared with np.gradient")
    print("Maximum difference in Ex:", diff_Ex)
    print("Maximum difference in Ey:", diff_Ey)
    print()

    # HW2: Calculate the field yourself and fractional error
    true_Ex, true_Ey = total_field(charges, xgrid, ygrid)

    error_Ex = fractional_error(my_Ex, true_Ex)
    error_Ey = fractional_error(my_Ey, true_Ey)

    plot_fractional_error(error_Ex, 'HW2: Fractional Error in Ex')
    plot_fractional_error(error_Ey, 'HW2: Fractional Error in Ey')

    print("HW2: Fractional error")
    print("Max |fractional error Ex|:", np.nanmax(np.abs(error_Ex)))
    print("Max |fractional error Ey|:", np.nanmax(np.abs(error_Ey)))
    print("These errors are much larger than machine precision, about 1e-16.")
    print()

    # HW3(a): Launch from positive charge toward negative charge
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (positive.x + 0.8, positive.y, 'withfield', negative)
        ],
        title='HW3(a): Line from Positive Charge Toward Negative Charge'
    )

    # HW3(b): Launch from negative charge toward positive charge
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (negative.x - 0.8, negative.y, 'againstfield', positive)
        ],
        title='HW3(b): Line from Negative Charge Toward Positive Charge'
    )

    # HW3(c): Launch from positive charge perpendicular to dipole axis
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (positive.x, positive.y + 0.8, 'withfield', negative)
        ],
        title='HW3(c): Line from Positive Charge Perpendicular to Dipole Axis'
    )   

    # HW3(d): Reflection
    print("HW3(d):")
    print("A field line is not the same as the trajectory of a charged particle.")
    print("A field line only shows the direction of the electric field at each point.")
    print("A particle trajectory depends on force, acceleration, mass, charge, and initial velocity.")
    print()

    # HW4: Final field line figure
    start_points_hw4 = make_start_points_from_charge(
        source_charge=positive,
        target_charge=negative,
        number_of_lines=16,
        launch_radius=0.8
    )

    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=start_points_hw4,
        title='HW4: Dipole Field Lines'
    )

    plt.show()


if __name__ == "__main__":
    main()