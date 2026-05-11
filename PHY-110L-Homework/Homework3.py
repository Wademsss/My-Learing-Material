# PHY 110L Homework 3
# Yucheng Liu
# 919908992
# AI clarifies: In this project, GitHub Colpiolt was used as a code completion tool.

import numpy as np
import matplotlib.pyplot as plt

### Grid setup
def make_grid(n=51, xmin=-25, xmax=25, ymin=-25, ymax=25):
    coords = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    ygrid = coords[0, :, :]
    xgrid = coords[1, :, :]
    return xgrid, ygrid


### Charge class
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

### Physics helpers

## HW1 Part
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

## HW2 Part
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

## HW3 Part
def unit_field(charges, x, y):
    Ex, Ey = total_field(charges, x, y)
    mag = np.hypot(Ex, Ey)

    if mag == 0 or not np.isfinite(mag):
        return None

    return Ex / mag, Ey / mag


def unit_equipotential_direction(charges, x, y):
    direction = unit_field(charges, x, y)

    if direction is None:
        return None

    ux, uy = direction

    # Rotate E direction by 90 degrees
    return -uy, ux

# Euler and RK2 steps
def euler_step(charges, x, y, h, direction='withfield'):
    direction_vector = unit_field(charges, x, y)

    if direction_vector is None:
        return None

    ux, uy = direction_vector

    if direction == 'againstfield':
        ux = -ux
        uy = -uy

    return x + h * ux, y + h * uy

def rk2_step(charges, x, y, h, direction='withfield'):
    direction_vector = unit_field(charges, x, y)

    if direction_vector is None:
        return None

    ux1, uy1 = direction_vector

    if direction == 'againstfield':
        ux1 = -ux1
        uy1 = -uy1

    # Midpoint
    xm = x + 0.5 * h * ux1
    ym = y + 0.5 * h * uy1

    mid_direction = unit_field(charges, xm, ym)

    if mid_direction is None:
        return None

    ux2, uy2 = mid_direction

    if direction == 'againstfield':
        ux2 = -ux2
        uy2 = -uy2

    return x + h * ux2, y + h * uy2

def rk2_equipotential_step(charges, x, y, h):
    direction_vector = unit_equipotential_direction(charges, x, y)

    if direction_vector is None:
        return None

    ux1, uy1 = direction_vector

    xm = x + 0.5 * h * ux1
    ym = y + 0.5 * h * uy1

    mid_direction = unit_equipotential_direction(charges, xm, ym)

    if mid_direction is None:
        return None

    ux2, uy2 = mid_direction

    return x + h * ux2, y + h * uy2

### Field line helpers
def trace_field_line(charges, x0, y0, stop_x, stop_y, step=0.35, stop_radius=0.8, max_steps=3000, direction='withfield'):
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

### Plot helpers
## HW1 Part
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
    
## HW2 Part
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


def plot_field_lines(charges, positive, negative, start_points, title):
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

    plt.scatter(positive.x, positive.y, color='red', s=220, edgecolors='black', zorder=5)

    plt.scatter(negative.x, negative.y, color='blue', s=220, edgecolors='black', zorder=5)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()

def make_start_points_from_charge(source_charge, target_charge, number_of_lines, launch_radius, angle_offset):
    start_points = []

    angles = np.linspace(0, 2*np.pi, number_of_lines, endpoint=False)
    angles = angles + angle_offset

    for theta in angles:
        x0 = source_charge.x + launch_radius * np.cos(theta)
        y0 = source_charge.y + launch_radius * np.sin(theta)

        if source_charge.q > 0:
            direction = 'withfield'
        else:
            direction = 'againstfield'

        start_points.append((x0, y0, direction, target_charge))

    return start_points

### Main
def main():

    # Set up
    xgrid, ygrid = make_grid()

    positive = Charge(-7, 0, +1)
    negative = Charge(7, 0, -1)
    charges = [positive, negative]

    V = total_potential(charges, xgrid, ygrid)

    # HW1
    my_Ex, my_Ey = my_centered_difference_field(V)
    np_Ex, np_Ey = electric_field_from_potential(V)

    interior = np.s_[1:-1, 1:-1]

    diff_Ex = np.nanmax(np.abs(my_Ex[interior] - np_Ex[interior]))
    diff_Ey = np.nanmax(np.abs(my_Ey[interior] - np_Ey[interior]))

    print("HW1: Centered difference compared with np.gradient")
    print("Maximum difference in Ex:", diff_Ex)
    print("Maximum difference in Ey:", diff_Ey)
    print()

    # HW2
    true_Ex, true_Ey = total_field(charges, xgrid, ygrid)

    error_Ex = fractional_error(my_Ex, true_Ex)
    error_Ey = fractional_error(my_Ey, true_Ey)

    plot_fractional_error(error_Ex, 'HW2: Fractional Error in Ex')
    plot_fractional_error(error_Ey, 'HW2: Fractional Error in Ey')

    print("HW2: Fractional error")
    print("Max fractional error Ex:", np.nanmax(np.abs(error_Ex)))
    print("Max fractional error Ey:", np.nanmax(np.abs(error_Ey)))
    print("These errors are much larger than machine precision, about 1e-16.")
    print()

    # HW3(a)
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (positive.x + 0.8, positive.y, 'withfield', negative)
        ],
        title='HW3(a): Line from Positive Charge Toward Negative Charge'
    )

    # HW3(b)
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (negative.x - 0.8, negative.y, 'againstfield', positive)
        ],
        title='HW3(b): Line from Negative Charge Toward Positive Charge'
    )

    # HW3(c)
    plot_field_lines(
        charges,
        positive,
        negative,
        start_points=[
            (positive.x, positive.y + 0.8, 'withfield', negative)
        ],
        title='HW3(c): Line from Positive Charge Perpendicular to Dipole Axis'
    )   

    # HW3(d)

    # Field lines are geometric curves that show the direction of the electric field, while the 
    # trajectory of a charged particle is determined by Newton’s law F=qE=ma and depends on the 
    # particle’s mass, charge, and initial velocity, so a particle does not generally move along 
    # a field line.


    # HW4
    # Positive charge
    positive_start_points = make_start_points_from_charge(
        source_charge=positive,
        target_charge=negative,
        number_of_lines=8,
        launch_radius=0.8,
        angle_offset=np.pi / 8
    )

    # Negative charge
    negative_start_points = [
        (negative.x + 0.8, negative.y + 0.4, 'againstfield', positive),
        (negative.x + 0.8, negative.y - 0.4, 'againstfield', positive),
    ]

    start_points_hw4 = positive_start_points + negative_start_points

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