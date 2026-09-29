"""
Replay the paper's Example-2 truth trajectory (Table 3) in your MAV viewer.

    python animation_paper.py            # animated follow-camera flight
    python animation_paper.py --static   # whole trajectory (compare with paper Fig. 1)
    python animation_paper.py --save     # write paper_flight.gif
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import paper_trajectory as PT
from mav_viewer import MavViewer, MsgState

ap = argparse.ArgumentParser()
ap.add_argument("--static", action="store_true")
ap.add_argument("--save", action="store_true")
ap.add_argument("--speed", type=float, default=20.0, help="sim seconds per real second")
args = ap.parse_args()

t, S = PT.generate()                       # 0.04 s steps, 1000 s

if args.static:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(S[:, 1], S[:, 0], -S[:, 2], lw=2)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Height [m]")
    ax.set_title("True trajectory (compare with paper Fig. 1)")
    plt.show()
    raise SystemExit

frame_dt = 0.5                              # one animation frame every 0.5 s of sim time
idx = np.arange(0, len(t), int(round(frame_dt / (t[1] - t[0]))))
viewer_states = [MsgState(pn=S[i, 0], pe=S[i, 1], pd=S[i, 2],
                          phi=S[i, 6], theta=S[i, 7], psi=S[i, 8]) for i in idx]

viewer = MavViewer(elev=22, azim=-125)
if hasattr(viewer, "scale"):
    viewer.scale = 3.0
FOLLOW = 60.0


def frame_fn(j):
    st = viewer_states[j]
    viewer.fixed_limits = (
        (st.pn - FOLLOW, st.pn + FOLLOW),
        (st.pe - FOLLOW, st.pe + FOLLOW),
        (st.pd - FOLLOW, st.pd + FOLLOW),
    )
    if j == 0 and hasattr(viewer, "reset_trail"):
        viewer.reset_trail()
    i = idx[j]
    viewer.update(st,
                  title=f"t={t[i]:6.1f}s  V={S[i,3]:.1f} m/s  h={-S[i,2]:.0f} m  "
                        f"roll={np.degrees(S[i,6]):+.1f}  pitch={np.degrees(S[i,7]):+.1f}  "
                        f"hdg={np.degrees(S[i,8]):+.0f} deg")
    return viewer.poly,


anim = FuncAnimation(viewer.fig, frame_fn, frames=len(idx),
                     interval=1000 * frame_dt / args.speed, repeat=False)

if args.save:
    anim.save("paper_flight.gif", writer="pillow", fps=20)
else:
    plt.show()