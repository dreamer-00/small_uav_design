import numpy as np


# ============================================================
# GRAVITY
# ============================================================

g = 9.81


def gravity_force(phi, theta, mass):
    """
    Gravity force expressed in the BODY frame.

    Body-frame convention:

        x -> forward
        y -> right
        z -> down

    Returns:

        [fx, fy, fz] in Newtons
    """

    fx = -mass * g * np.sin(theta)

    fy = mass * g * np.cos(theta) * np.sin(phi)

    fz = mass * g * np.cos(theta) * np.cos(phi)

    return np.array([
        fx,
        fy,
        fz
    ])


# ============================================================
# AIR DATA
# ============================================================

def air_data(u, v, w):
    """
    Calculate airspeed, angle of attack, and sideslip angle.

    Inputs:
        u, v, w : body-frame air-relative velocity components

    Returns:
        [Va, alpha, beta]

        Va    : airspeed [m/s]
        alpha : angle of attack [rad]
        beta  : sideslip angle [rad]
    """

    # Airspeed magnitude
    Va = np.sqrt(
        u**2 +
        v**2 +
        w**2
    )

    # Avoid division by zero at zero airspeed
    if Va < 1e-8:

        alpha = 0.0
        beta = 0.0

    else:

        # Angle of attack
        alpha = np.arctan2(w, u)

        # Sideslip angle
        ratio = v / Va

        # Numerical safety
        ratio = np.clip(ratio, -1.0, 1.0)

        beta = np.arcsin(ratio)

    return np.array([
        Va,
        alpha,
        beta
    ])
