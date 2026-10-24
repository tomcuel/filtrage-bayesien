import kalman
import numpy as np
import graph_helper as gh
import matplotlib.pyplot as plt

np.random.seed(1)

N = 9

XTrue = np.zeros((N + 1, 2))
XTrue[0] = np.array([20.,
                     12.])
Y = np.zeros((N+1, 1))
XEst = np.zeros((N + 1, 2))
PEst = np.zeros((N + 1, 2, 2))

X0 = np.array([0.,
               10.])
P0 = np.array([[100.,    0.],
               [0.,    1.]])

XEst[0] = X0
PEst[0] = P0

DeltaT = 1

kf = kalman.KalmanFilter(X0, P0)

def F(k):
    return np.array([[1., DeltaT],
                     [0,  1.]])

def Q(k):
    return np.array([[1., 0.],
                     [0, 0.001]])

def H(k):
    return np.array([[1., 0.]])
def R(k):
    return np.array([[1.]])


def simulate_x_step(k):
    global XTrue
    Fk = F(k)
    Qk = Q(k)
    noise = np.linalg.cholesky(Qk) @ np.random.randn(2)
    XTrue[k+1] = (
        Fk @ XTrue[k]
        + noise
    )

def simulate_y(k):
    global XTrue, Y
    Hk = H(k)
    Rk = R(k)
    Y[k] = Hk @ XTrue[k] + np.linalg.cholesky(Rk) @ np.random.randn(1)

gains = []
for k in range(N):
    simulate_x_step(k)
    if k == 0:
        XTrue[k+1] = np.array([30.83,12.07])    # valeur imposée par la slide

    kf.predict(F(k), Q(k))
    if k == 0:
        print("x(1|0) ",kf.x)
        print("P(1|0) ",kf.P)

    simulate_y(k+1)
    if k == 0:
        Y[k+1] = 29.91                          # mesure imposée par la slide

    innovation, S, K = kf.update(Y[k+1], H(k+1), R(k+1))
    gains.append(K.ravel())
    if k == 0:
        print("innovation ",innovation, " S ", S)
        print("K ",K.ravel())
        print("x(1|1) ",kf.x)
        print("P(1|1) ",kf.P)
        print("----------------")

    XEst[k+1] = kf.x
    PEst[k+1] = kf.P

print("Gains K_k (position, vitesse) :\n", np.round(np.array(gains), 4))


fig, axes = plt.subplots(2, 1, sharex=True)

t = np.arange(N+1) - 1

position_true = XTrue[:,0]
velocity_true = XTrue[:,1]
sigma_position = PEst[:,0,0] ** 0.5
sigma_velocity = PEst[:,1,1] ** 0.5

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
plt.savefig("figures/tp1_part0a.png", dpi=130)
plt.show()
