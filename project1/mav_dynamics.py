import numpy as np


# ============================================================
# ROTATION MATRICES
# ============================================================

def rotation_x(phi):
    c = np.cos(phi)
    s = np.sin(phi)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(psi):
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


# ============================================================
# EQUATION (3.14)
# POSITION KINEMATICS
# ============================================================

def position_derivative(phi, theta, psi, u, v, w):

    velocity_body = np.array([u, v, w])

    R = (
        rotation_z(psi)
        @ rotation_y(theta)
        @ rotation_x(phi)
    )

    velocity_inertial = R @ velocity_body

    return velocity_inertial


# ============================================================
# EQUATION (3.15)
# TRANSLATIONAL DYNAMICS
# ============================================================

def velocity_derivative(
    u, v, w,
    p, q, r,
    fx, fy, fz,
    mass
):

    du = r * v - q * w + fx / mass

    dv = p * w - r * u + fy / mass

    dw = q * u - p * v + fz / mass

    return np.array([du, dv, dw])


# ============================================================
# EQUATION (3.16)
# EULER ANGLE KINEMATICS
# ============================================================

def euler_rates(phi, theta, p, q, r):

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    tantheta = np.tan(theta)
    costheta = np.cos(theta)

    transform = np.array([
        [
            1,
            sinphi * tantheta,
            cosphi * tantheta
        ],

        [
            0,
            cosphi,
            -sinphi
        ],

        [
            0,
            sinphi / costheta,
            cosphi / costheta
        ]
    ])

    body_rates = np.array([p, q, r])

    euler_rate = transform @ body_rates

    return euler_rate


# ============================================================
# EQUATION (3.11)
# ROTATIONAL DYNAMICS
# ============================================================

def angular_rate_derivative(
    p, q, r,
    l, m_moment, n,
    J
):

    omega = np.array([p, q, r])

    moment = np.array([
        l,
        m_moment,
        n
    ])

    # Angular momentum
    h = J @ omega

    # Rotating-frame coupling term
    omega_cross_h = np.cross(omega, h)

    # J * omega_dot
    rhs = moment - omega_cross_h

    # For learning, we explicitly use J^-1
    J_inv = np.linalg.inv(J)

    omega_dot = J_inv @ rhs

    return omega_dot


# ============================================================
# COMPLETE 12-STATE EQUATION
# ============================================================

def mav_dynamics(state, forces, moments, mass, J):

    # --------------------------------------------------------
    # Unpack the 12 states
    # --------------------------------------------------------

    (
        pn, pe, pd,
        u, v, w,
        phi, theta, psi,
        p, q, r
    ) = state

    # --------------------------------------------------------
    # Unpack forces
    # --------------------------------------------------------

    fx, fy, fz = forces

    # --------------------------------------------------------
    # Unpack moments
    # --------------------------------------------------------

    l, m_moment, n = moments

    # --------------------------------------------------------
    # Equation (3.14)
    # Position derivatives
    # --------------------------------------------------------

    pos_dot = position_derivative(
        phi,
        theta,
        psi,
        u,
        v,
        w
    )

    # --------------------------------------------------------
    # Equation (3.15)
    # Velocity derivatives
    # --------------------------------------------------------

    vel_dot = velocity_derivative(
        u,
        v,
        w,
        p,
        q,
        r,
        fx,
        fy,
        fz,
        mass
    )

    # --------------------------------------------------------
    # Equation (3.16)
    # Euler-angle derivatives
    # --------------------------------------------------------

    angle_dot = euler_rates(
        phi,
        theta,
        p,
        q,
        r
    )

    # --------------------------------------------------------
    # Equation (3.17)
    # Angular-rate derivatives
    # --------------------------------------------------------

    rate_dot = angular_rate_derivative(
        p,
        q,
        r,
        l,
        m_moment,
        n,
        J
    )

    # --------------------------------------------------------
    # Combine everything into the 12-state derivative
    # --------------------------------------------------------

    state_dot = np.concatenate([
        pos_dot,
        vel_dot,
        angle_dot,
        rate_dot
    ])

    return state_dot


# ============================================================
# APPENDIX E PARAMETERS
# ============================================================

mass = 11.0

Jx = 0.824
Jy = 1.135
Jz = 1.759
Jxz = 0.120

J = np.array([
    [Jx, 0.0, -Jxz],
    [0.0, Jy, 0.0],
    [-Jxz, 0.0, Jz]
])