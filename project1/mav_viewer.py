import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MsgState:
    def __init__(self, pn=0.0, pe=0.0, pd=0.0, phi=0.0, theta=0.0, psi=0.0):
        self.pn=pn
        self.pe=pe
        self.pd=pd
        self.phi=phi
        self.theta=theta
        self.psi=psi
def rotation_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rotation_y(theta):
    c, s=np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rotation_z(psi):
    c, s=np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def _body_points(
    fuse_l1=2.8,
    fuse_l2=1.0,
    fuse_l3=6.0,
    fuse_w=1.0,
    fuse_h=1.0,
    wing_l=2.5,
    wing_w=8.0,
    tailwing_l=1.0,
    tailwing_w=3.0,
    tail_h=1.5
):
    return np.array([
    [fuse_l1,            0.0,             0.0          ],  # 1: nose tip
    [fuse_l2,            fuse_w / 2,     -fuse_h / 2    ],  # 2: nose-box top-right... (see below)
    [fuse_l2,           -fuse_w / 2,     -fuse_h / 2    ],  # 3
    [fuse_l2,           -fuse_w / 2,      fuse_h / 2    ],  # 4
    [fuse_l2,            fuse_w / 2,      fuse_h / 2    ],  # 5
    [-fuse_l3,            0.0,             0.0          ],  # 6: tail tip
    [0.0,                 wing_w / 2,      0.0          ],  # 7: wing leading edge, right tip
    [-wing_l,             wing_w / 2,      0.0          ],  # 8: wing trailing edge, right tip
    [-wing_l,            -wing_w / 2,      0.0          ],  # 9: wing trailing edge, left tip
    [0.0,                -wing_w / 2,      0.0          ],  # 10: wing leading edge, left tip
    [-(fuse_l3 - tailwing_l), tailwing_w / 2, 0.0        ],  # 11
    [-fuse_l3,             tailwing_w / 2,  0.0          ],  # 12
    [-fuse_l3,            -tailwing_w / 2,  0.0          ],  # 13
    [-(fuse_l3 - tailwing_l), -tailwing_w / 2, 0.0        ],  # 14
    [-(fuse_l3 - tailwing_l), 0.0,           0.0          ],  # 15
    [-fuse_l3,             0.0,            -tail_h        ],  # 16: vertical tail tip
]).T
_FACES_1INDEXED=[[1,2,3], [1, 3, 4], [1, 4, 5], [1, 5, 2], [2, 3, 6], [3, 4, 6], [4, 5, 6], [5, 2, 6], [7, 8, 9, 10], [11, 12, 13, 14], [6, 15, 16]]
_FACE_COLORS=["0.85", "0.85", "0.85", "0.85", "0.6", "0.6", "0.6", "0.6", "steelblue", "indianred", "indianred"]
def _make_mesh(points_3xN, faces_1indexed):
    mesh=[]
    for face in faces_1indexed:
        col_indices=[p-1 for p in face]
        mesh.append(points_3xN[:, col_indices].T)
    return mesh
class MavViewer:
    def __init__(self, fixed_limits=None, elev=20, azim=-40):
        self.points_body=_body_points()
        self.fixed_limits=fixed_limits
        self.poly=None
        self.fig=plt.figure(figsize=(7,7))
        self.ax=self.fig.add_subplot(111, projection="3d")
        self.elev, self.azim=elev, azim
        self._style_axes()
    def _style_axes(self):
        self.ax.set_xlabel("x (North)")
        self.ax.set_ylabel("y (East)")
        self.ax.set_zlabel("z (Down)")
        self.ax.grid(True)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.set_alpha(0.5)
        self.ax.view_init(elev=self.elev, azim=self.azim)
    def _world_points(self, state:MsgState):
        R=rotation_z(state.psi) @ rotation_y(state.theta) @ rotation_x(state.phi)
        rotated= R @ self.points_body
        position=np.array([state.pn, state.pe, state.pd]).reshape(3, 1)
        return rotated+position
    def update(self, state: MsgState, title=None):
        world_points=self._world_points(state)
        mesh=_make_mesh(world_points, _FACES_1INDEXED)
        if self.poly is None:
            self.poly=Poly3DCollection(mesh, facecolors=_FACE_COLORS, edgecolors="k", linewidths=0.5)
            self.ax.add_collection3d(self.poly)
        else:
            self.poly.set_verts(mesh)
        if self.fixed_limits is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.fixed_limits
        else:
            pad=3.0
            xmin, ymin, zmin = world_points.min(axis=1)-pad
            xmax, ymax, zmax = world_points.max(axis=1)+pad
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)
        if title:
            self.ax.set_title(title)
    @staticmethod
    def compute_fixed_limits(states, pad=3.0):
        dummy=MavViewer.__new__(MavViewer)
        dummy.points_body=_body_points()
        all_pts=[dummy._world_points(s) for s in states]
        all_pts=np.concatenate(all_pts, axis=1)
        xmin, ymin, zmin = all_pts.min(axis=1) - pad
        xmax, ymax, zmax = all_pts.max(axis=1) + pad
        return (xmin, xmax), (ymin, ymax), (zmin, zmax)
        

