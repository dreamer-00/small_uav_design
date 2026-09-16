import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mav_viewer import MavViewer, MsgState
from mav_dynamics import mav_dynamics, mass, J
dt=0.10
n_frames=60
state=np.zeros(12)
forces=np.array([11.0, 0.0, 0.0])
moments=np.array([0.0, 0.0, 0.0])
states=[]
times=np.arange(n_frames)*dt
for i in range(n_frames):
    states.append(state.copy())
    state_dot = mav_dynamics(
        state,
        forces,
        moments,
        mass,
        J
    )
    state = state + state_dot * dt

viewer_states=[]
for state in states:
    viewer_state=MsgState(pn=state[0], pe=state[1], pd=state[2], phi=state[6], theta=state[7], psi=state[8])
    viewer_states.append(viewer_state)
fixed_limits=MavViewer.compute_fixed_limits(viewer_states)
viewer=MavViewer(fixed_limits=fixed_limits, elev=22, azim=-40)
def frame_fn(i):
    state=viewer_states[i]
    viewer.update(state, title=f"t={times[i]:.2f}s")
    return viewer.poly
anim=FuncAnimation(viewer.fig, frame_fn, frames=n_frames, interval=dt*1000)
plt.show()
