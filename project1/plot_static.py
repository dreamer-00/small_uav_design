"""
One static figure of the whole Example-2 trajectory with aircraft "snapshots"
drawn along the path (like a screenshot of the animation).

    python plot_static.py            # show + save static_trajectory.png
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import paper_trajectory as PT
from mav_viewer import _FACES, _shade, rotation_body_to_ned

# ---- snapshots: (time [s], label)
SNAPS = [(0, "start"), (225, "left turn"), (400, "right turn"),
         (590, "climb"), (760, "descent"), (1000, "end")]
AIRCRAFT_SCALE = 28.0        # visual size multiplier for the 8 m wingspan model
BOX = (1.0, 0.9, 0.35)      # display aspect ratio of (East, North, Height) axes

t, S = PT.generate()
E, N, Hh = S[:, 1], S[:, 0], -S[:, 2]                       # plot as East, North, Height

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

# ---- axis limits (data units)
pad = 300
xr = (E.min() - pad, E.max() + pad)
yr = (N.min() - pad, N.max() + pad)
zr = (0.0, Hh.max() + 60)
ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
ax.set_box_aspect(BOX)

# ---- trajectory coloured by time, plus ground shadow
pts = np.column_stack([E, N, Hh])[::25]
segs = np.stack([pts[:-1], pts[1:]], axis=1)
lc = Line3DCollection(segs, cmap="viridis", linewidth=2.5)
lc.set_array(t[::25][:-1])
ax.add_collection3d(lc)
ax.plot(E, N, np.full_like(E, zr[0]), color="0.6", lw=1, zorder=0)
cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02, location="bottom", aspect=40)
cb.set_label("time [s]")

# ---- aircraft snapshots (anisotropic scaling so the model looks undistorted)
disp_per_data = np.array([BOX[0] / (xr[1] - xr[0]),
                          BOX[1] / (yr[1] - yr[0]),
                          BOX[2] / (zr[1] - zr[0])])
q = disp_per_data[0] * AIRCRAFT_SCALE                       # display size of 1 body metre
k = q / disp_per_data                                       # data units per body metre, per axis

for ts, label in SNAPS:
    i = min(int(round(ts / (t[1] - t[0]))), len(t) - 1)
    R = rotation_body_to_ned(S[i, 6], S[i, 7], S[i, 8])
    centre = np.array([E[i], N[i], Hh[i]])
    mesh, cols = [], []
    for v, c in _FACES:
        ned = (R @ v.T).T                                    # body -> NED offsets [m]
        off = np.column_stack([ned[:, 1], ned[:, 0], -ned[:, 2]]) * k
        world_ned = ned + np.array([S[i, 0], S[i, 1], S[i, 2]])
        mesh.append(off + centre)
        cols.append(_shade(world_ned, c))
    ax.add_collection3d(Poly3DCollection(mesh, facecolors=cols, edgecolors="k", linewidths=0.3))
    ax.text(E[i], N[i], Hh[i] + 70, f"{label}\n{ts:g} s", ha={"end": "left", "descent": "right"}.get(label, "center"), fontsize=8)

ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Height [m]")
ax.set_title("Example 2: true aircraft trajectory (Table 3 maneuvers)")
ax.view_init(elev=28, azim=-50)

fig.savefig("static_trajectory.png", dpi=160, bbox_inches="tight")
plt.show()