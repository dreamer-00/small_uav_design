import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# STATE USED BY THE VIEWER  (unchanged)
# ============================================================

class MsgState:
    def __init__(self, pn=0.0, pe=0.0, pd=0.0, phi=0.0, theta=0.0, psi=0.0):
        self.pn, self.pe, self.pd = pn, pe, pd
        self.phi, self.theta, self.psi = phi, theta, psi


# ============================================================
# ROTATION MATRICES  (unchanged)
# ============================================================

def rotation_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_body_to_ned(phi, theta, psi):
    return rotation_z(psi) @ rotation_y(theta) @ rotation_x(phi)


# ============================================================
# AIRCRAFT MESH  (body frame: x forward, y right, z down, metres)
# Each face is (Nx3 vertex array, base colour).
# ============================================================

def _lerp(a, b, s):
    return a + (b - a) * s


def _build_faces():
    faces = []

    # ---------------- fuselage: nose cone -> section A -> section B -> tail cone
    nose = np.array([3.0, 0.0, 0.0])
    tail = np.array([-6.5, 0.0, 0.0])
    # corner order: top-right, top-left, bottom-left, bottom-right  (z is DOWN, so top = -z)
    A = np.array([[1.4, .55, -.45], [1.4, -.55, -.45], [1.4, -.55, .45], [1.4, .55, .45]])
    B = np.array([[-2.5, .35, -.30], [-2.5, -.35, -.30], [-2.5, -.35, .30], [-2.5, .35, .30]])

    top, side, bottom = "0.88", "0.72", "0.50"
    side_col = [side, side, bottom, side]           # per edge: top, left, bottom, right
    edge_col = [top, side, bottom, side]
    for i in range(4):
        j = (i + 1) % 4
        faces.append((np.array([nose, A[i], A[j]]), edge_col[i]))
        faces.append((np.array([A[i], A[j], B[j], B[i]]), edge_col[i]))
        faces.append((np.array([B[i], B[j], tail]), edge_col[i]))

    # ---------------- main wing (right half, mirrored for left)
    rLE, tLE = np.array([0.9, 0.55, 0.0]), np.array([-0.1, 4.0, 0.0])
    rTE, tTE = np.array([-1.6, 0.55, 0.0]), np.array([-1.0, 4.0, 0.0])
    s_tip = (3.5 - 0.55) / (4.0 - 0.55)             # outer 0.5 m is coloured
    mLE, mTE = _lerp(rLE, tLE, s_tip), _lerp(rTE, tTE, s_tip)

    def mirror(v):
        return v * np.array([1, -1, 1])

    faces.append((np.array([rLE, mLE, mTE, rTE]), "steelblue"))
    faces.append((np.array([mirror(v) for v in (rLE, mLE, mTE, rTE)]), "steelblue"))
    faces.append((np.array([mLE, tLE, tTE, mTE]), "limegreen"))                       # right tip
    faces.append((np.array([mirror(v) for v in (mLE, tLE, tTE, mTE)]), "red"))        # left tip

    # ---------------- horizontal tail
    hr = np.array([[-5.0, .35, 0], [-5.9, 1.7, 0], [-6.5, 1.7, 0], [-6.2, .35, 0]])
    faces.append((hr, "indianred"))
    faces.append((hr * np.array([1, -1, 1]), "indianred"))

    # ---------------- vertical fin
    fin = np.array([[-4.6, 0, -.3], [-5.7, 0, -1.8], [-6.5, 0, -1.8], [-6.3, 0, -.2]])
    faces.append((fin, "firebrick"))

    # ---------------- propeller disc (translucent)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    disc = np.column_stack([np.full_like(ang, 3.05), 0.9 * np.cos(ang), 0.9 * np.sin(ang)])
    faces.append((disc, (0.15, 0.15, 0.15, 0.35)))

    return faces


_FACES = _build_faces()
_LIGHT = np.array([0.3, 0.2, -1.0])
_LIGHT /= np.linalg.norm(_LIGHT)


def _world_faces(state, scale=1.0):
    R = rotation_body_to_ned(state.phi, state.theta, state.psi)
    pos = np.array([state.pn, state.pe, state.pd])
    return [((R @ (scale * v).T).T + pos, col) for v, col in _FACES]


def _shade(verts, col):
    rgba = np.array(to_rgba(col))
    n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    nn = np.linalg.norm(n)
    k = 0.6 + 0.4 * abs(n @ _LIGHT) / nn if nn > 1e-12 else 0.8
    rgba[:3] = np.clip(rgba[:3] * k, 0, 1)
    return rgba


# ============================================================
# MAV VIEWER
# ============================================================

class MavViewer:
    """
    fixed_limits : ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
    follow       : half-width [m] of a cubic window that follows the aircraft (overrides limits)
    scale        : visual scale factor of the aircraft model
    trail        : draw the flown path
    """

    def __init__(self, fixed_limits=None, elev=20, azim=-40,
                 scale=1.0, follow=None, trail=True, figsize=(8, 8)):
        self.fixed_limits = fixed_limits
        self.follow = follow
        self.scale = scale
        self.elev, self.azim = elev, azim
        self.poly = None
        self.use_trail = trail
        self._trail = []

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.trail_line, = self.ax.plot([], [], [], color="crimson", lw=1.2)
        self._style_axes()

    def _style_axes(self):
        self.ax.set_xlabel("x (North) [m]")
        self.ax.set_ylabel("y (East) [m]")
        self.ax.set_zlabel("z (Down) [m]")
        self.ax.grid(True)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.set_alpha(0.4)
        self.ax.view_init(elev=self.elev, azim=self.azim)

    def reset_trail(self):
        self._trail = []

    def update(self, state, title=None):
        wf = _world_faces(state, self.scale)
        mesh = [v for v, _ in wf]
        colors = [_shade(v, c) for v, c in wf]

        if self.poly is None:
            self.poly = Poly3DCollection(mesh, facecolors=colors,
                                         edgecolors="k", linewidths=0.4)
            self.ax.add_collection3d(self.poly)
        else:
            self.poly.set_verts(mesh)
            self.poly.set_facecolor(colors)

        if self.use_trail:
            self._trail.append([state.pn, state.pe, state.pd])
            tr = np.array(self._trail)
            self.trail_line.set_data(tr[:, 0], tr[:, 1])
            self.trail_line.set_3d_properties(tr[:, 2])

        if self.follow is not None:
            c = np.array([state.pn, state.pe, state.pd])
            (xmin, ymin, zmin), (xmax, ymax, zmax) = c - self.follow, c + self.follow
        elif self.fixed_limits is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.fixed_limits
        else:
            allp = np.concatenate([v for v, _ in wf])
            xmin, ymin, zmin = allp.min(axis=0) - 3.0
            xmax, ymax, zmax = allp.max(axis=0) + 3.0

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmax, zmin)                 # reversed: +Down points down
        self.ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

        if title:
            self.ax.set_title(title)

    @staticmethod
    def compute_fixed_limits(states, pad=3.0, scale=1.0):
        pts = np.concatenate([np.concatenate([v for v, _ in _world_faces(s, scale)])
                              for s in states])
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        return ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]))