"""
SINS/GPS loosely-coupled simulation (Example 2 of Liu et al., ISA Trans. 2018) with a UKF.

Frames : nav = ENU (x east, y north, z up); body = RFU (x right, y forward, z up)
Attitude: C_b^n from (heading psi [CCW+], pitch theta, roll gamma), standard Chinese SINS convention
States  : x = [phi(3), dv(3), dp(3) (E,N,U in metres), eps(3), nabla(3)]   (15)
Meas.   : z = SINS - GPS = [dv(3), dp(3)] + noise                          (6)
Loop    : closed-loop (feedback) - attitude/vel/pos errors reset after each update,
          gyro/accel bias estimates are kept and subtracted from IMU data.

Usage:  python sins_gps_ukf.py --mc 3 --noise gaussian
        python sins_gps_ukf.py --mc 3 --noise mixture
        python sins_gps_ukf.py --check          (noise-free sanity test)
"""
import argparse
import numpy as np

D2R = np.pi / 180.0
RE, E2 = 6378137.0, 0.0066943799901413
WIE = 7.292115e-5
G0 = 9.78
DT, T_END, GPS_EVERY = 0.04, 1000.0, 25          # 25 Hz IMU, 1 Hz GPS
GN = np.array([0.0, 0.0, -G0])


# ----------------------------------------------------------------- helpers
def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def radii(L):
    s2 = np.sin(L) ** 2
    return RE * (1 - E2) / (1 - E2 * s2) ** 1.5, RE / np.sqrt(1 - E2 * s2)   # RM, RN


def w_ie(L):
    return WIE * np.array([0.0, np.cos(L), np.sin(L)])


def w_en(v, L, H):
    RM, RN = radii(L)
    return np.array([-v[1] / (RM + H), v[0] / (RN + H), v[0] * np.tan(L) / (RN + H)])


def dcm(psi, th, ga):
    sp, cp, st, ct, sg, cg = np.sin(psi), np.cos(psi), np.sin(th), np.cos(th), np.sin(ga), np.cos(ga)
    return np.array([[cg * cp - sg * st * sp, -ct * sp, sg * cp + cg * st * sp],
                     [cg * sp + sg * st * cp, ct * cp, sg * sp - cg * st * cp],
                     [-sg * ct, st, cg * ct]])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    a = vee(R - R.T) / 2.0
    s, c = np.linalg.norm(a), (np.trace(R) - 1) / 2.0
    th = np.arctan2(s, c)
    return a if s < 1e-14 else a * th / s


def orthonorm(C):
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt


# ----------------------------------------------------------------- truth (Table 3)
def make_truth():
    N = int(round(T_END / DT))
    t = np.arange(N + 1) * DT
    V = np.interp(t, [0, 90, 100, 895, 900, 1000], [5, 5, 10, 10, 5, 5])
    roll = D2R * np.interp(t, [0, 200, 205, 250, 255, 355, 360, 450, 455, 1000],
                           [0, 0, -2.04, -2.04, 0, 0, 1.02, 1.02, 0, 0])
    pitch = D2R * np.interp(t, [0, 555, 565, 615, 625, 725, 735, 785, 795, 1000],
                            [0, 0, 20, 20, 0, 0, -20, -20, 0, 0])
    psi = np.zeros(N + 1)
    for k in range(N):                                   # coordinated turn (left turn = +psi)
        psi[k + 1] = psi[k] - DT * G0 * np.tan(roll[k]) / V[k]

    C = np.array([dcm(psi[k], pitch[k], roll[k]) for k in range(N + 1)])
    v = np.einsum('kij,j->ki', C, np.array([0, 1.0, 0])) * V[:, None]
    pos = np.zeros((N + 1, 3))                           # L, lambda, H
    pos[0] = [45.779 * D2R, 126.670 * D2R, 100.0]
    for k in range(N):
        L, _, H = pos[k]
        RM, RN = radii(L)
        pos[k + 1] = pos[k] + DT * np.array([v[k, 1] / (RM + H),
                                             v[k, 0] / ((RN + H) * np.cos(L)), v[k, 2]])
    # ideal IMU outputs, consistent with the discrete mechanization below
    wib = np.zeros((N, 3))
    fb = np.zeros((N, 3))
    for k in range(N):
        L, _, H = pos[k]
        wnb = log_so3(C[k].T @ C[k + 1]) / DT
        wien = w_ie(L) + w_en(v[k], L, H)
        wib[k] = C[k].T @ wien + wnb
        fn = (v[k + 1] - v[k]) / DT + np.cross(w_ie(L) * 2 + w_en(v[k], L, H), v[k]) - GN
        fb[k] = C[k].T @ fn
    return dict(N=N, t=t, C=C, v=v, pos=pos, wib=wib, fb=fb)


# ----------------------------------------------------------------- error model (psi-angle, ENU)
def error_F(C, v, L, H, fb):
    """Continuous-time F for x = [phi, dv, dp(E,N,U m), eps, nabla]."""
    RM, RN = radii(L)
    wie, wen = w_ie(L), w_en(v, L, H)
    tL, cL = np.tan(L), np.cos(L)
    fn = C @ fb
    F = np.zeros((15, 15))

    # d(omega_en)/d(dv) and d(omega_ie), d(omega_en)/d(dp_N)   (dL = dp_N / RM)
    Wen_v = np.array([[0, -1 / RM, 0], [1 / RN, 0, 0], [tL / RN, 0, 0]])
    Wie_p = np.zeros((3, 3)); Wie_p[:, 1] = WIE * np.array([0, -np.sin(L), cL]) / RM
    Wen_p = np.zeros((3, 3)); Wen_p[2, 1] = v[0] / (RN * cL ** 2) / RM

    F[0:3, 0:3] = -skew(wie + wen)
    F[0:3, 3:6] = Wen_v
    F[0:3, 6:9] = Wie_p + Wen_p
    F[0:3, 9:12] = -C
    F[3:6, 0:3] = skew(fn)
    F[3:6, 3:6] = -skew(2 * wie + wen) + skew(v) @ Wen_v
    F[3:6, 6:9] = skew(v) @ (2 * Wie_p + Wen_p)
    F[3:6, 12:15] = C
    F[6:9, 3:6] = np.eye(3)
    F[6, 7] = v[0] * tL / RM
    return F


# ----------------------------------------------------------------- UKF
class UKF:
    def __init__(self, n, alpha=1.0, beta=0.0, kappa=0.0):
        self.n = n
        lam = alpha ** 2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 0.5 / self.c)
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c
        self.Wc[0] = lam / self.c + (1 - alpha ** 2 + beta)

    def sigma(self, x, P):
        try:
            S = np.linalg.cholesky(self.c * P)
        except np.linalg.LinAlgError:                     # numerical PD loss (UKF weakness)
            S = np.linalg.cholesky(self.c * (P + 1e-12 * np.trace(P) / self.n * np.eye(self.n)))
        return np.vstack([x, x + S.T, x - S.T])

    def predict(self, x, P, f_pts, Q):
        Xp = f_pts(self.sigma(x, P))                     # f_pts maps (2n+1,n) -> (2n+1,n)
        xp = self.Wm @ Xp
        d = Xp - xp
        Pp = (self.Wc[:, None] * d).T @ d + Q
        return xp, 0.5 * (Pp + Pp.T)

    def update(self, x, P, z, h_pts, R):
        X = self.sigma(x, P)
        Y = h_pts(X)
        yp = self.Wm @ Y
        dy, dx = Y - yp, X - x
        Pyy = (self.Wc[:, None] * dy).T @ dy + R
        Pxy = (self.Wc[:, None] * dx).T @ dy
        K = np.linalg.solve(Pyy.T, Pxy.T).T
        xn = x + K @ (z - yp)
        Pn = P - K @ Pyy @ K.T
        return xn, 0.5 * (Pn + Pn.T)


# ----------------------------------------------------------------- one Monte-Carlo run
SIG_R = np.array([0.1, 0.1, 0.1, 10.0, 10.0, 1.0])          # nominal GPS noise (m/s, m)
SIG_R_BIG = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 10.0])   # outlier component
ARW = 0.02 * D2R / 60.0                                      # 0.02 deg/sqrt(h) -> rad/sqrt(s)
VRW = 5e-5 * G0                                              # 5e-5 g -> (m/s^2)/sqrt(Hz)
EPS_TRUE = np.full(3, 0.05 * D2R / 3600.0)                   # 0.05 deg/h
NAB_TRUE = np.full(3, 1e-4 * G0)                             # 1e-4 g
H_MAT = np.hstack([np.zeros((6, 3)), np.eye(6), np.zeros((6, 6))])


def run_once(T, rng, noise="gaussian", sensors=True, init_err=True):
    N = T["N"]
    P = np.diag(np.r_[np.array([1, 1, 5]) * D2R, [2, 2, 2], [100, 100, 10],
                      np.full(3, 0.5 * D2R / 3600), np.full(3, 1e-3 * G0)] ** 2)
    Qd = np.zeros((15, 15))
    Qd[0:3, 0:3] = np.eye(3) * ARW ** 2 * DT
    Qd[3:6, 3:6] = np.eye(3) * VRW ** 2 * DT
    R = np.diag(SIG_R ** 2)
    ukf = UKF(15)

    # initial SINS state = truth + initial errors
    C = T["C"][0].copy(); v = T["v"][0].copy(); pos = T["pos"][0].copy()
    if init_err:
        phi0 = np.array([0.1, 0.1, 0.5]) * D2R
        C = (np.eye(3) - skew(phi0)) @ C
        v = v + 0.2
        RM, RN = radii(pos[0])
        pos = pos + np.array([10 / RM, 10 / (RN * np.cos(pos[0])), 1.0])
    x = np.zeros(15)
    errs = []

    for k in range(N):
        # ---- IMU
        wib, fb = T["wib"][k].copy(), T["fb"][k].copy()
        if sensors:
            wib = wib + EPS_TRUE + rng.standard_normal(3) * ARW / np.sqrt(DT)
            fb = fb + NAB_TRUE + rng.standard_normal(3) * VRW / np.sqrt(DT)
        wib_c, fb_c = wib - x[9:12], fb - x[12:15]           # bias compensation (feedback)
        L, lam, H = pos

        # ---- filter predict (f is linear in the error state -> Phi; sigma points still used)
        F = error_F(C, v, L, H, fb_c)
        A = F * DT
        Phi = np.eye(15) + A + 0.5 * A @ A
        x, P = ukf.predict(x, P, lambda X: X @ Phi.T, Qd)

        # ---- SINS mechanization
        RM, RN = radii(L)
        wie, wen = w_ie(L), w_en(v, L, H)
        wnb = wib_c - C.T @ (wie + wen)
        acc = C @ fb_c - np.cross(2 * wie + wen, v) + GN
        pos = pos + DT * np.array([v[1] / (RM + H), v[0] / ((RN + H) * np.cos(L)), v[2]])
        v = v + DT * acc
        C = C @ exp_so3(wnb * DT)
        if k % 50 == 0:
            C = orthonorm(C)

        # ---- GPS update, 1 Hz
        if (k + 1) % GPS_EVERY == 0:
            L, lam, H = pos
            RM, RN = radii(L)
            tv, tp = T["v"][k + 1], T["pos"][k + 1]
            d_sins = np.r_[v - tv,
                           (RN * np.cos(L) * (lam - tp[1])), (RM * (L - tp[0])), (H - tp[2])]
            if not sensors:
                r = np.zeros(6)
            elif noise == "gaussian":
                r = SIG_R * rng.standard_normal(6)
            else:
                big = rng.random(6) < 0.1
                r = np.where(big, SIG_R_BIG, SIG_R) * rng.standard_normal(6)
            z = d_sins - r                                    # SINS - GPS
            x, P = ukf.update(x, P, z, lambda X: X @ H_MAT.T, R)

            # ---- closed-loop correction, then reset the first 9 states
            C = orthonorm((np.eye(3) + skew(x[0:3])) @ C)
            v = v - x[3:6]
            pos = pos - np.array([x[7] / RM, x[6] / (RN * np.cos(L)), x[8]])
            x[0:9] = 0.0

            # corrected-solution error (paper order: vE vN vU, L(N) lam(E) H)
            L2, lam2, H2 = pos
            RM2, RN2 = radii(L2)
            errs.append(np.r_[v - tv, RM2 * (L2 - tp[0]),
                              RN2 * np.cos(L2) * (lam2 - tp[1]), H2 - tp[2]])
    return np.array(errs)                                     # (K, 6)


def summarize(E, k1=11, k2=1000):
    """E: (M, K, 6) -> RMSE(k) (48) and TRMSE (49)."""
    rmse = np.sqrt(np.mean(E ** 2, axis=0))
    trmse = np.sqrt(np.mean(rmse[k1 - 1:k2] ** 2, axis=0))
    return rmse, trmse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=3, help="Monte-Carlo runs (paper: 100)")
    ap.add_argument("--noise", choices=["gaussian", "mixture"], default="gaussian")
    ap.add_argument("--check", action="store_true", help="noise-free sanity test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    T = make_truth()
    print(f"truth built: {T['N']} steps, final H = {T['pos'][-1, 2]:.1f} m")
    if a.check:
        E = run_once(T, np.random.default_rng(0), sensors=False, init_err=False)
        print("noise-free, zero init error -> max |error| per channel:", np.abs(E).max(axis=0))
    else:
        rng = np.random.default_rng(a.seed)
        E = np.array([run_once(T, rng, a.noise) for _ in range(a.mc)])
        _, tr = summarize(E)
        names = ["vE m/s", "vN m/s", "vU m/s", "L m", "lam m", "H m"]
        print(f"UKF, {a.noise} noise, {a.mc} runs, TRMSE:")
        for n_, val in zip(names, tr):
            print(f"  {n_:7s} {val:.4f}")