# PHY 110L Homework 3
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Colpiolt was used as a code completion tool.

import numpy as np
import matplotlib.pyplot as plt




# ============================================================
# Charge class
# ============================================================
class Charge:
    def __init__(self, x, y, q):
        self.x = float(x)
        self.y = float(y)
        self.q = float(q)

    def potential(self, x, y):
        r = np.hypot(x - self.x, y - self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.q / r

    def field(self, x, y):
        dx = x - self.x
        dy = y - self.y
        r = np.hypot(dx, dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            ex = self.q * dx / r**3
            ey = self.q * dy / r**3
        return ex, ey

# ============================================================
# Physics helpers
# ============================================================

# HW1 Part
def total_potential(charges, x, y):
    total = np.zeros_like(np.asarray(x), dtype=float)
    for charge in charges:
        total += charge.potential(x, y)
    return total

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

## HW2 Part
def total_field(charges, x, y):
    ex_total = 0.0
    ey_total = 0.0
    for charge in charges:
        ex, ey = charge.field(x, y)
        ex_total += ex
        ey_total += ey
    return ex_total, ey_total

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

## HW3 Part
def unit_field(charges, x, y):
    ex, ey = total_field(charges, x, y)
    mag = np.hypot(ex, ey)
    if mag == 0 or not np.isfinite(mag):
        return None
    return ex / mag, ey / mag

def apply_direction(vector, direction):
    ux, uy = vector
    if direction == 'againstfield':
        return -ux, -uy
    return ux, uy

def unit_equipotential_direction(charges, x, y):
    field_direction = unit_field(charges, x, y)
    if field_direction is None:
        return None

    ux, uy = field_direction
    return -uy, ux       # rotate electric field direction by 90 degrees

# Euler and RK2 steps
def euler_step(charges, x, y, h, direction='withfield'):
    vector = unit_field(charges, x, y)
    if vector is None:
        return None

    ux, uy = apply_direction(vector, direction)
    return x + h * ux, y + h * uy


def rk2_step(charges, x, y, h, direction='withfield'):
    vector = unit_field(charges, x, y)
    if vector is None:
        return None

    ux1, uy1 = apply_direction(vector, direction)

    # Use a half Euler step to estimate the midpoint.
    xm = x + 0.5 * h * ux1
    ym = y + 0.5 * h * uy1

    midpoint_vector = unit_field(charges, xm, ym)
    if midpoint_vector is None:
        return None

    ux2, uy2 = apply_direction(midpoint_vector, direction)
    return x + h * ux2, y + h * uy2


def rk2_equipotential_step(charges, x, y, h):
    vector = unit_equipotential_direction(charges, x, y)
    if vector is None:
        return None

    ux1, uy1 = vector

    xm = x + 0.5 * h * ux1
    ym = y + 0.5 * h * uy1

    midpoint_vector = unit_equipotential_direction(charges, xm, ym)
    if midpoint_vector is None:
        return None

    ux2, uy2 = midpoint_vector
    return x + h * ux2, y + h * uy2

# ============================================================
# Starting points
# ============================================================
def make_start_points_from_charge(source_charge, number_of_lines=8, launch_radius=0.8, angle_offset=0):
    start_points = []
    angles = np.linspace(0, 2 * np.pi, number_of_lines, endpoint=False) + angle_offset

    for theta in angles:
        x0 = source_charge.x + launch_radius * np.cos(theta)
        y0 = source_charge.y + launch_radius * np.sin(theta)
        direction = 'withfield' if source_charge.q > 0 else 'againstfield'
        start_points.append((x0, y0, direction))

    return start_points

def find_equipotential_start_points(charges, levels, x_min=-6.5, x_max=6.5, y=1.0, num_points=1000):
    xline = np.linspace(x_min, x_max, num_points)
    potentials = total_potential(charges, xline, y)
    start_points = []

    for level in levels:
        above = potentials > level
        if np.any(above):
            index = np.argmax(above)
            start_points.append((xline[index], y, level))

    return start_points

# ============================================================
# Drawing helpers
# ============================================================
def setup_axes(ax, title, bounds=(-25, 25, -25, 25)):
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

def trace_field_line(
    ax,
    charges,
    x0,
    y0,
    h=0.3,
    method='rk2',
    direction='withfield',
    stop_charge=None,
    stop_radius=0.8,
    max_steps=4000,
    bounds=(-25, 25, -25, 25),
    color='black',
    linewidth=1.0,
):
    x = float(x0)
    y = float(y0)
    xmin, xmax, ymin, ymax = bounds

    for _ in range(max_steps):
        if method == 'euler':
            next_point = euler_step(charges, x, y, h, direction)
        else:
            next_point = rk2_step(charges, x, y, h, direction)

        if next_point is None:
            break

        x_new, y_new = next_point
        ax.plot([x, x_new], [y, y_new], color=color, linewidth=linewidth)

        x, y = x_new, y_new

        if stop_charge is not None:
            if np.hypot(x - stop_charge.x, y - stop_charge.y) < stop_radius:
                break
        
        margin = 50
        
        if (
            x < xmin - 50 or
            x > xmax + 50 or
            y < ymin - 50 or
            y > ymax + 50
        ):
            break



def launch_lines_from_charge( charges, source_charge, target_charge, number_of_lines=8, launch_radius=0.8):
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

        trace_field_line(charges, x0, y0, target_charge.x, target_charge.y, direction=direction)

def trace_equipotential(
    ax,
    charges,
    x0,
    y0,
    h=0.25,
    max_steps=5000,
    start_check_steps=20,
    close_radius=0.4,
    bounds=(-25, 25, -25, 25),
    color='orange',
    linewidth=1.2,
):
    x = float(x0)
    y = float(y0)
    xmin, xmax, ymin, ymax = bounds

    for step_index in range(max_steps):
        next_point = rk2_equipotential_step(charges, x, y, h)
        if next_point is None:
            break

        x_new, y_new = next_point

        outside = x_new < xmin or x_new > xmax or y_new < ymin or y_new > ymax
        if outside:
            # Re-enter from the opposite side without drawing a long artificial line across the plot.
            x = np.clip(x_new, xmin, xmax)
            y = np.clip(y_new, ymin, ymax)
            continue

        ax.plot([x, x_new], [y, y_new], color=color, linewidth=linewidth)
        x, y = x_new, y_new

        if step_index > start_check_steps and np.hypot(x - x0, y - y0) < close_radius:
            break

def draw_field_lines(ax, charges, start_points, h=0.3, method='rk2', stop_charge=None, color='black'):
    for x0, y0, direction in start_points:
        trace_field_line(
            ax,
            charges,
            x0,
            y0,
            h=h,
            method=method,
            direction=direction,
            stop_charge=stop_charge,
            color=color,
        )

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

def plot_charges(ax, charges):
    for charge in charges:
        color = 'red' if charge.q > 0 else 'blue'
        ax.scatter(charge.x, charge.y, color=color, s=240, edgecolors='black', zorder=5)

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

def plot_field_lines(charges, start_points, title, h=0.3, method='rk2', stop_charge=None, color='black'):
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_field_lines(ax, charges, start_points, h=h, method=method, stop_charge=stop_charge, color=color)
    plot_charges(ax, charges)
    setup_axes(ax, title)
    fig.tight_layout()

# Unit 3 part
def plot_field_lines_and_equipotentials(dipole, positive, negative):
    fig, ax = plt.subplots(figsize=(8, 8))

    start_points = make_start_points_from_charge(
        positive,
        number_of_lines=12,
        launch_radius=0.8,
        angle_offset=np.pi / 12,
    )
    draw_field_lines(ax, dipole, start_points, h=0.25, method='rk2', stop_charge=negative, color='black')

    levels = np.linspace(-0.25, 0.25, 9)
    levels = [level for level in levels if abs(level) > 1e-6]

    equipotential_starts = find_equipotential_start_points(
        dipole,
        levels,
        x_min=-6.5,
        x_max=6.5,
        y=1.0,
    )

    for x0, y0, _ in equipotential_starts:
        trace_equipotential(ax, dipole, x0=x0, y0=y0, h=0.25, color='orange')

    plot_charges(ax, dipole)
    setup_axes(ax, 'HW4: Field Lines and Equipotentials')
    fig.tight_layout()

def plot_euler_vs_rk2(dipole, positive, negative, start_points, h, title):
    fig, ax = plt.subplots(figsize=(7, 6))

    draw_field_lines(ax, dipole, start_points, h=h, method='euler', stop_charge=negative, color='gray')
    draw_field_lines(ax, dipole, start_points, h=h, method='rk2', stop_charge=negative, color='black')

    plot_charges(ax, dipole)
    setup_axes(ax, title)
    fig.tight_layout()

def plot_one_equipotential(dipole, positive, negative, start_points):
    fig, ax = plt.subplots(figsize=(8, 8))

    draw_field_lines(ax, dipole, start_points, h=0.25, method='rk2', stop_charge=negative, color='black')
    trace_equipotential(ax, dipole, x0=1, y0=0, h=0.25, color='orange')

    plot_charges(ax, dipole)
    setup_axes(ax, 'HW3: One Equipotential')
    fig.tight_layout()

### Main
def main():

    # Set up
    positive = Charge(-7, 0, +1)
    negative = Charge(7, 0, -1)
    dipole = [positive, negative]

    start_points = make_start_points_from_charge(
        positive,
        number_of_lines=8,
        launch_radius=0.8,
        angle_offset=np.pi / 8,
    )
    # HW1(a), HW1(b): Euler vs RK2
    plot_euler_vs_rk2(dipole, positive, negative, start_points, h=0.5,
                      title='HW1(a): Euler Gray vs RK2 Black, h = 0.5')
    plot_euler_vs_rk2(dipole, positive, negative, start_points, h=0.1,
                      title='HW1(b): Euler Gray vs RK2 Black, h = 0.1')

    # HW2(a): single positive charge
    single_positive = [Charge(0, 0, +1)]
    start_single = make_start_points_from_charge(single_positive[0], number_of_lines=12, launch_radius=0.8)
    plot_field_lines(single_positive, start_single, title='HW2(a): Single Positive Charge',
                     h=0.3, method='rk2', stop_charge=None, color='black')

    # HW2(b): two positive charges
    pos1 = Charge(-7, 0, +1)
    pos2 = Charge(7, 0, +1)
    two_positive = [pos1, pos2]
    start_two_positive = (
        make_start_points_from_charge(pos1, number_of_lines=8, launch_radius=0.8)
        + make_start_points_from_charge(pos2, number_of_lines=8, launch_radius=0.8)
    )
    plot_field_lines(two_positive, start_two_positive, title='HW2(b): Two Positive Charges',
                     h=0.3, method='rk2', stop_charge=None, color='black')

    # HW2(c): two positive charges and one negative charge
    p1 = Charge(-7, 0, +1)
    p2 = Charge(7, 0, +1)
    n1 = Charge(0, 8, -1)
    mixed = [p1, p2, n1]
    start_mixed = (
        make_start_points_from_charge(p1, number_of_lines=8, launch_radius=0.8)
        + make_start_points_from_charge(p2, number_of_lines=8, launch_radius=0.8)
    )
    plot_field_lines(mixed, start_mixed, title='HW2(c): Two Positive Charges and One Negative Charge',
                     h=0.3, method='rk2', stop_charge=n1, color='black')

    # HW3, HW4
    plot_one_equipotential(dipole, positive, negative, start_points)
    plot_field_lines_and_equipotentials(dipole, positive, negative)

    print('HW1(c):')
    print('For h = 0.5, Euler and RK2 visibly differ. RK2 follows the curved field lines more accurately.')
    print('For h = 0.1, Euler improves, but RK2 still gives better accuracy for similar visual smoothness.')
    print('This confirms that RK2 has better accuracy than Euler without simply using many smaller steps.')
    print()

    print('HW2 verification:')
    print('A single positive charge produces field lines going radially outward to infinity.')
    print('Two positive charges produce lines going outward, bending away from the region between the charges.')
    print('Two positive charges with one negative charge produce some lines ending on the negative charge and others going to infinity.')

    plt.show()



if __name__ == "__main__":
    main()