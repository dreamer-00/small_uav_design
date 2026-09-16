import numpy as np
def rotation_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rotation_y(theta):
    c, s=np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rotation_z(psi):
    c, s=np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def position_derivative(phi, theta, psi, u, v, w):
    velocity_body=np.array([u, v, w])
    R=rotation_z(psi) @ rotation_y(theta) @ rotation_x(phi)
    velocity_inertial= R @ velocity_body
    return velocity_inertial
def velocity_derivative(u, v, w, p, q, r, fx, fy, fz, mass):
    force=np.array([fx, fy, fz])
    du=r*v-q*w+(fx/mass)
    dv=p*w-r*u+(fy/mass)
    dw=q*u-p*v+(fz/mass)
    return np.array([du, dv, dw])
def euler_rates(phi, theta, p, q, r):
    sinphi=np.sin(phi)
    tantheta=np.tan(theta)
    cosphi=np.cos(phi)
    costheta=np.cos(theta)
    transform=np.array([[1, sinphi*tantheta, cosphi*tantheta], [0, cosphi, -sinphi], [0, sinphi/costheta, cosphi/costheta]])
    position=np.array([p, q, r])
    euler_mat= transform @ position
    return euler_mat
def angular_rate_derivative(p, q, r, l, m_moment, n, J):
    omega=np.array([p, q, r])
    moment=np.array([l, m_moment, n])
    h= J @ omega
    omega_cross_h=np.cross(omega, h)
    rhs=moment-omega_cross_h
    J_inv=np.linalg.inv(J)
    omega_dot=J_inv @ rhs
    return omega_dot
def mav_dynamics(state, force, moments, mass, J):
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    fx, fy, fz= force
    l, m_moment, n = moments
    pos_dot=position_derivative(phi, theta, psi, u, v, w)
    vel_dot=velocity_derivative(u, v, w, p, q, r, fx, fy, fz, mass)
    angle_dot=euler_rates(phi, theta, p, q, r)
    rate_dot=angular_rate_derivative(p, q, r, l, m_moment, n, J)
    state_dot=np.concatenate([pos_dot, vel_dot, angle_dot, rate_dot])
    return state_dot
Jx = 0.824
Jy = 1.135
Jz = 1.759
Jxz = 0.120
J = np.array([
    [Jx, 0.0, -Jxz],
    [0.0, Jy, 0.0],
    [-Jxz, 0.0, Jz]
])
mass=11.0
