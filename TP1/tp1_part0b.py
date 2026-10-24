import ekf as ekf
import numpy as np
import graph_helper as gh;
import matplotlib.pyplot as plt

np.random.seed(1)

N = 30


SensorPos = np.array([0.0, -20.0])

XTrue = np.zeros((N + 1, 2))
XTrue[0] = np.array([2.,
                     4.])
Y = np.zeros((N+1, 1))
XEst = np.zeros((N + 1, 2))
PEst = np.zeros((N + 1, 2, 2))

X0 = np.array([0.,
               5])
P0 = np.array([[100.,    0.],
               [0.,    1.]])

XEst[0] = X0
PEst[0] = P0

DeltaT = 1

ekf = ekf.ExtendedKalmanFilter(X0, P0)

F = np.array([[1., DeltaT],
                 [0,  1.]])

Q = np.array([[0.0001, 0.],
              [0, 0.01]])

R = np.array([
    [np.deg2rad(1.0)**2]
])


def f(x):
    return F @ x

def F_jacobian(x):
    return F

def h_bearing(x):
    xs, ys = SensorPos
    s = x[0]
    angle = np.arctan2(-ys, s - xs)
    return np.array([angle])

def H_bearing(x):
    xs, ys = SensorPos
    s = x[0]
    den = (s - xs)**2 + ys**2
    return np.array([[ys / den, 0.0]])

def wrap_angle_innovation(innovation):
    innovation = innovation.copy()
    innovation[0] = (innovation[0] + np.pi) % (2 * np.pi) - np.pi
    return innovation

def simulate_x_step(k):
    global XTrue
    noise = np.linalg.cholesky(Q) @ np.random.randn(2)
    XTrue[k+1] = f(XTrue[k]) + noise

def simulate_y(k):
    global XTrue, Y
    Y[k] = h_bearing(XTrue[k]) + np.linalg.cholesky(R) @ np.random.randn(1)

for k in range(N):
    simulate_x_step(k)
    ekf.predict(f,F_jacobian, Q)
    simulate_y(k+1)
    innovation, S, K = ekf.update(Y[k+1], h_bearing, H_bearing, R)
    if k == 0:
        print("innovation ",innovation)
        print("K ",K)
        print("x(k+1|k+1) ",ekf.x)
        print("P(k+1|k+1) ",ekf.P)
        print("----------------")

    XEst[k+1] = ekf.x
    PEst[k+1] = ekf.P


fig, axes = plt.subplots(2, 1, sharex=True)

t = np.arange(N+1) - 1

position_true = XTrue[:,0]
velocity_true = XTrue[:,1]
sigma2_position = PEst[:,0,0]
sigma2_velocity = PEst[:,1,1]
sigma_position = sigma2_position ** 0.5
sigma_velocity = sigma2_velocity ** 0.5

print("sigma_position ", sigma_position)
print("sigma_velocity ", sigma_velocity)

position_est = XEst[:,0]
velocity_est = XEst[:,1]


gh.plot_with_interval(
    t,
    curves=[
        {"y": position_true, "label": "position réelle", "color": "blue"},
        {"y": position_est,  "label": "position estimée", "color": "red"}
    ],
    lower=position_est - 3 * sigma_position,
    upper=position_est + 3 * sigma_position,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[0]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": velocity_true, "label": "vitesse réelle", "color": "blue"},
        {"y": velocity_est,  "label": "vitesse estimée", "color": "red"}
    ],
    lower=velocity_est - 3 * sigma_velocity,
    upper=velocity_est + 3 * sigma_velocity,
    interval_label=r"incertitude ($3\sigma$)",
    xlabel="temps (s)",
    ylabel="vitesse (m/s)",
    ax=axes[1]
)

plt.tight_layout()
plt.savefig("figures/tp1_part0b.png")
plt.show()
