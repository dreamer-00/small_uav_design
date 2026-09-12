import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mav_viewer import MavViewer, MsgState
def trajectory(t):
    phi=np.deg2rad(21)*np.sin(2*np.pi*0.25*t)
    theta=np.deg2rad(11)*np.sin(2*np.pi*0.1*t)
    psi=np.deg2rad(21)*np.sin(2*np.pi*0.15*t)
    pn=3*t
    pe=2*np.sin(2*np.pi*0.1*t)
    pd=-1*t
    return MsgState(pn=pn, pe=pe, pd=pd, phi=phi, theta=theta, psi=psi)
n_frames=100
dt=0.15
times=np.arange(n_frames)*dt
all_states=[trajectory(t) for t in times]
fixed_limits=MavViewer.compute_fixed_limits(all_states)
viewer=MavViewer(fixed_limits=fixed_limits)
def frame_fn(i):
    state=all_states[i]
    viewer.update(state, title=f"t={times[i]:.2f}s")
    return viewer.poly, 
anim=FuncAnimation(viewer.fig, frame_fn, frames=n_frames, interval=dt*1000)
plt.show()