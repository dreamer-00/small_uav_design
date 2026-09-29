"""
Truth trajectory for Example 2 (Table 3), in BEARD & McLAIN conventions:
    NED navigation frame, body x-forward / y-right / z-down, 3-2-1 Euler angles.

Returns the same 12-state layout your mav_dynamics.py uses:
    [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]

This is a KINEMATIC script (no forces/aerodynamics): speed, roll and pitch are
prescribed by Table 3, heading follows the coordinated-turn relation
    psi_dot = g * tan(phi) / V
"""
import numpy as np
import paper_params as P


def generate(dt=P.DT_IMU, t_end=P.T_END):
    t = np.arange(0.0, t_end + dt / 2, dt)
    n = len(t)

    V = np.interp(t, *P.SPEED_KNOTS)
    phi = np.interp(t, P.ROLL_KNOTS[0], np.array(P.ROLL_KNOTS[1]) * P.D2R)
    theta = np.interp(t, P.PITCH_KNOTS[0], np.array(P.PITCH_KNOTS[1]) * P.D2R)

    # heading from coordinated turn (Beard: right roll -> psi increases)
    psi = np.zeros(n)
    for k in range(n - 1):
        psi[k + 1] = psi[k] + dt * P.G * np.tan(phi[k]) / V[k]

    # inertial velocity = R_b^i [V, 0, 0]  (first column of Rz Ry Rx)
    vn = V * np.cos(theta) * np.cos(psi)
    ve = V * np.cos(theta) * np.sin(psi)
    vd = -V * np.sin(theta)

    pn = np.concatenate([[0.0], np.cumsum(vn[:-1]) * dt])
    pe = np.concatenate([[0.0], np.cumsum(ve[:-1]) * dt])
    pd = -P.H0 + np.concatenate([[0.0], np.cumsum(vd[:-1]) * dt])

    # body angular rates from Euler rates (inverse of Beard eq. 3.16)
    phid, thetad, psid = (np.gradient(a, dt) for a in (phi, theta, psi))
    p = phid - psid * np.sin(theta)
    q = thetad * np.cos(phi) + psid * np.sin(phi) * np.cos(theta)
    r = -thetad * np.sin(phi) + psid * np.cos(phi) * np.cos(theta)

    states = np.column_stack([pn, pe, pd, V, np.zeros(n), np.zeros(n),
                              phi, theta, psi, p, q, r])
    return t, states


if __name__ == "__main__":
    t, S = generate()
    print(f"{len(t)} samples, dt={t[1]-t[0]:.2f}s")
    print(f"final pn={S[-1,0]:.1f} m, pe={S[-1,1]:.1f} m, height={-S[-1,2]:.1f} m, "
          f"psi={np.degrees(S[-1,8]):.1f} deg")
    print(f"max height={-S[:,2].min():.1f} m, max |p,q,r| deg/s="
          f"{np.degrees(np.abs(S[:,9:]).max(axis=0)).round(2)}")