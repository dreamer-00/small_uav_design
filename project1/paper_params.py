"""
All numbers from Liu et al., ISA Trans. 80 (2018), Example 2 (SINS/GPS).
Everything paper-specific lives here so the rest of your code stays generic.
"""
import numpy as np

D2R = np.pi / 180.0
G = 9.81                      # m/s^2 (paper doesn't state g; Beard uses 9.81)

# ---------------------------------------------------------------- timing
DT_IMU = 0.04                 # SINS / IMU period [s]
DT_GPS = 1.0                  # GPS period [s]
T_END = 1000.0                # total simulation time [s]
N_MC = 100                    # Monte Carlo runs in the paper

# ---------------------------------------------------------------- initial condition
LAT0 = 45.779 * D2R           # [rad]
LON0 = 126.670 * D2R          # [rad]
H0 = 100.0                    # [m]  -> pd0 = -100 in Beard's NED
V0_NED = np.array([0.0, 5.0, 0.0])   # paper writes v^n = [0 5 0] in ENU: 5 m/s to the NORTH

# ---------------------------------------------------------------- Table 3 maneuvers
# (times [s], values) as piecewise-linear knots
SPEED_KNOTS = ([0, 90, 100, 895, 900, 1000],
               [5, 5, 10, 10, 5, 5])                                   # m/s
ROLL_KNOTS = ([0, 200, 205, 250, 255, 355, 360, 450, 455, 1000],
              [0, 0, -2.04, -2.04, 0, 0, 1.02, 1.02, 0, 0])            # deg (negative = left turn)
PITCH_KNOTS = ([0, 555, 565, 615, 625, 725, 735, 785, 795, 1000],
               [0, 0, 20, 20, 0, 0, -20, -20, 0, 0])                   # deg

# ---------------------------------------------------------------- IMU errors (used in the sensor chapter)
GYRO_BIAS = np.full(3, 0.05 * D2R / 3600.0)      # 0.05 deg/h  -> rad/s
ACCEL_BIAS = np.full(3, 1e-4 * G)                # 1e-4 g      -> m/s^2
GYRO_ARW = 0.02 * D2R / 60.0                     # 0.02 deg/sqrt(h) -> rad/sqrt(s)
ACCEL_VRW = 5e-5 * G                             # 5e-5 g (used as a noise density)

# ---------------------------------------------------------------- GPS noise, std devs [vE vN vU m/s, N E m, U m]
GPS_SIGMA = np.array([0.1, 0.1, 0.1, 10.0, 10.0, 1.0])
GPS_SIGMA_OUTLIER = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 10.0])   # 10 % mixture component
OUTLIER_PROB = 0.1

# ---------------------------------------------------------------- initial errors and filter init
INIT_ATT_ERR = np.array([0.1, 0.1, 0.5]) * D2R    # rad
INIT_VEL_ERR = np.array([0.2, 0.2, 0.2])          # m/s
INIT_POS_ERR = np.array([10.0, 10.0, 1.0])        # m (paper: (1800/(Re*pi)) deg = 10 m)
P0_DIAG = np.r_[np.array([1, 1, 5]) * D2R, [2, 2, 2], [100, 100, 10],
                np.full(3, 0.5 * D2R / 3600.0), np.full(3, 1e-3 * G)] ** 2