from math import sin, cos, atan2, pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ---- Helper functions ----

# Display error ellipses
def plot_covariance_ellipse(xEst, PEst, axes, lineType):
    """
    Plot one covariance ellipse from covariance matrix
    """

    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    if eigval[smallind] < 0:
        print('Pb with Pxy :\n', Pxy)
        exit()

    t = np.arange(0, 2 * pi + 0.1, 0.1)
    a = sqrt(eigval[bigind])
    b = sqrt(eigval[smallind])
    x = [3 * a * cos(it) for it in t]
    y = [3 * b * sin(it) for it in t]
    angle = atan2(eigvec[1, bigind], eigvec[0, bigind])     # vecteur propre = COLONNE de eigvec
    rot = np.array([[cos(angle), -sin(angle)],
                    [sin(angle), cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    axes.plot(px, py, lineType, lw=0.8)


# fit angle between -pi and pi
def angle_wrap(a):
    if (a > np.pi):
        a = a - 2 * pi
    elif (a < -np.pi):
        a = a + 2 * pi
    return a


# composes two transformations
def tcomp(tab, tbc):
    assert tab.ndim == 2
    assert tbc.ndim == 2

    result = tab[2, 0] + tbc[2, 0]

    result = angle_wrap(result)
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    tac = tab[0:2] + np.array([[c, -s], [s, c]]) @ tbc[0:2]
    tac = np.vstack((tac, result))

    return tac


# ---- Global variables (initialisées dans run_simulation) ----

nSteps = 6000
Map = None
xTrue = None
xOdom = None

# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.01, 0.01, 1*pi/180]) ** 2
PYTrue = np.diag([5.0, 6*pi/180]) ** 2      # le capteur mesure toujours distance ET direction


# ---- Simulator functions ----

# return true control at step k
def get_robot_control(k):
    global nSteps
    # generate  sin trajectory
    u = np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / nSteps)]]).T
    return u


# simulate new true robot position
def simulate_world(k):
    global xTrue
    u = get_robot_control(k)
    xTrue = tcomp(xTrue, u)
    xTrue[2, 0] = angle_wrap(xTrue[2, 0])


# computes and returns noisy odometry (toujours avec le VRAI bruit QTrue)
def get_odometry(k):
    global xOdom  # internal to robot low-level controller
    global QTrue

    u = get_robot_control(k)
    xnow = tcomp(xOdom, u)
    uNoise = np.sqrt(QTrue) @ np.random.randn(3)
    uNoise = np.array([uNoise]).T
    xnow = tcomp(xnow, uNoise)
    xOdom = xnow
    u = u + uNoise
    return xnow, u


# generate a noisy observation of a random feature (toujours avec le VRAI bruit PYTrue)
def get_observation(k, with_failure=False, partial_obs=None, failure_start=2500, failure_end=3500):
    # sensor failure mode
    if with_failure and (failure_start < k < failure_end):
        return [None, None]
    global Map, xTrue, PYTrue

    # tirages faits AVANT le test de panne : la suite aléatoire reste la même d'un scénario à l'autre
    iFeature = np.random.randint(0, Map.shape[1] - 1)
    zNoise = np.sqrt(PYTrue) @ np.random.randn(2)
    zNoise = np.array([zNoise]).T
    z = observation_model(xTrue, iFeature, Map) + zNoise      # mesure complète (distance, direction)
    z[1, 0] = angle_wrap(z[1, 0])
    if partial_obs == "distance":
        z = z[[0], :]
    elif partial_obs == "direction":
        z = z[[1], :]
    return [z, iFeature]


# ---- model functions ----

# evolution model (f)
def motion_model(x, u):
    xPred = tcomp(x, u)
    xPred[2, 0] = angle_wrap(xPred[2, 0])
    return xPred


# observation model (h)
def observation_model(xVeh, iFeature, Map, partial_obs=None):
    Delta = Map[0:2, iFeature:(iFeature+1)]-xVeh[0:2]
    dx = Delta[0, 0]
    dy = Delta[1, 0]
    if partial_obs == "distance":
        z = np.array([[np.linalg.norm(Delta)]])
        return z

    elif partial_obs == "direction":
        z = np.array([[atan2(dy, dx) - xVeh[2, 0]]])
        z[0, 0] = angle_wrap(z[0, 0])
        return z

    else:
        z = np.array([[np.linalg.norm(Delta)], [atan2(dy, dx) - xVeh[2, 0]]])
        z[1, 0] = angle_wrap(z[1, 0])
        return z


# ---- Jacobian functions ----

# h(x) Jacobian wrt x
def get_obs_jac(xPred, iFeature, Map, partial_obs=None):

    """
    dx = Map[0, iFeature] - xPred[0, 0]
    dy = Map[1, iFeature] - xPred[1, 0]
    q = dx**2 + dy**2
    r = sqrt(q)

    jac_distance = [-dx / r, -dy / r, 0]       # d r / d(x, y, theta)
    jac_direction = [dy / q, -dx / q, -1]      # d phi / d(x, y, theta)

    if partial_obs == "distance":
        jH = np.array([jac_distance])
    elif partial_obs == "direction":
        jH = np.array([jac_direction])
    else:
        jH = np.array([jac_distance, jac_direction])
    return jH
    """
    if partial_obs is None : 
        jH = np.zeros((2, 3))
        # -(x_k - x_t) / sqrt((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[0, 0] = -(Map[0, iFeature] - xPred[0, 0]) / sqrt((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        # (y_k - y_t) / sqrt((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[0, 1] = -(Map[1, iFeature] - xPred[1, 0]) / sqrt((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        jH[0, 2] = 0

        # (y_k - y_t) / ((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[1, 0] = (Map[1, iFeature] - xPred[1, 0]) / ((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        # (x_k - x_t) / ((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[1, 1] = -(Map[0, iFeature] - xPred[0, 0]) / ((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        jH[1, 2] = -1

    elif partial_obs == "distance":
        #print("distance")
        jH = np.zeros((1,3))
        jH[0, 0] = -(Map[0, iFeature] - xPred[0, 0]) / sqrt((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        # (y_k - y_t) / sqrt((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[0, 1] = -(Map[1, iFeature] - xPred[1, 0]) / sqrt((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        jH[0, 2] = 0

    elif partial_obs == "direction":
        #print("direction")
        jH = np.zeros((1,3))
        jH[0, 0] = (Map[1, iFeature] - xPred[1, 0]) / ((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        # (x_k - x_t) / ((x_k - x_t)^2 + (y_k - y_t)^2)
        jH[0, 1] = -(Map[0, iFeature] - xPred[0, 0]) / ((Map[0, iFeature] - xPred[0, 0])**2 + (Map[1, iFeature] - xPred[1, 0])**2)
        jH[0, 2] = -1

    return jH


# f(x,u) Jacobian wrt x
def A(x, u):
    Jac = np.zeros((3, 3))

    Jac[0, 0] = 1
    Jac[0, 1] = 0
    # - x_u * sin(theta) - y_u * cos(theta)
    Jac[0, 2] = -u[0, 0] * sin(x[2, 0]) - u[1, 0] * cos(x[2, 0])
    Jac[1, 0] = 0
    Jac[1, 1] = 1
    # x_u * cos(theta) - y_u * sin(theta)
    Jac[1, 2] = u[0, 0] * cos(x[2, 0]) - u[1, 0] * sin(x[2, 0])
    Jac[2, 0] = 0
    Jac[2, 1] = 0
    Jac[2, 2] = 1

    return Jac


# f(x,u) Jacobian wrt u
def B(x, u):
    Jac = np.zeros((3, 3))

    Jac[0, 0] = cos(x[2, 0])
    Jac[0, 1] = -sin(x[2, 0])
    Jac[0, 2] = 0
    Jac[1, 0] = sin(x[2, 0])
    Jac[1, 1] = cos(x[2, 0])
    Jac[1, 2] = 0
    Jac[2, 0] = 0
    Jac[2, 1] = 0
    Jac[2, 2] = 1

    return Jac


def run_simulation(partial_obs=None, with_failure=False, q_est=(10, 10, 10), py_est=(10, 10), seed=1, failure_start=2500, failure_end=3500):
    """
    partial_obs  : None (distance + direction), "distance" ou "direction"
    with_failure : panne capteur pour 2500 < k < 3500
    q_est        : QEst = diag(q_est) @ QTrue    -> bruit d'odométrie supposé PAR LE FILTRE
    py_est       : PYEst = diag(py_est) @ PYTrue -> bruit de perception supposé PAR LE FILTRE
    failure_start, failure_end : indices de début et fin de la panne capteur
    Le simulateur utilise toujours QTrue et PYTrue. Même graine => même carte, même trajectoire, mêmes bruits.
    """
    global Map, xTrue, xOdom
    np.random.seed(seed)

    # Location of landmarks
    Map = 140*np.random.rand(2, 30)-70

    # Modeled errors used in the Kalman filter process
    QEst = np.diag(q_est) @ QTrue
    PYEst = np.diag(py_est) @ PYTrue
    if partial_obs == "distance":
        PYEst = PYEst[[0]][:, [0]]
    elif partial_obs == "direction":
        PYEst = PYEst[[1]][:, [1]]

    # initial conditions
    xTrue = np.array([[1, -40, -pi/2]]).T
    xOdom = xTrue
    xEst = xTrue
    PEst = 100 * np.diag([1, 1, (1*pi/180)**2])

    # Init history (listes, empilées à la fin : beaucoup plus rapide que hstack à chaque pas)
    hxEst, hxTrue, hxOdom, hxP = [xEst], [xTrue], [xOdom], [PEst]

    for k in range(1, nSteps):

        # Simulate robot motion
        simulate_world(k)

        # Get odometry measurements
        xOdom, u = get_odometry(k)

        # Kalman prediction
        xPred = motion_model(xEst, u)  # function f
        PPred = A(xEst, u) @ PEst @ A(xEst, u).T + B(xEst, u) @ QEst @ B(xEst, u).T

        # Get random landmark observation
        [z, iFeature] = get_observation(k, with_failure=with_failure, partial_obs=partial_obs, failure_start=failure_start, failure_end=failure_end)

        if z is not None:
            # Predict observation
            zPred = observation_model(xPred, iFeature, Map, partial_obs)

            # get observation Jacobian
            H = get_obs_jac(xPred, iFeature, Map, partial_obs)

            # compute observation error (innovation)
            Innov = z - zPred
            if partial_obs is None:
                Innov[1, 0] = angle_wrap(Innov[1, 0])
            elif partial_obs == "direction":
                Innov[0, 0] = angle_wrap(Innov[0, 0])

            # compute Kalman gain
            S = H @ PPred @ H.T + PYEst
            W = PPred @ H.T @ np.linalg.inv(S)

            # perform kalman update
            xEst = xPred + W @ Innov
            xEst[2, 0] = angle_wrap(xEst[2, 0])

            PEst = PPred - W @ H @ PPred
            PEst = 0.5 * (PEst + PEst.T)  # ensure symetry

        else:
            # there was no observation available
            xEst = xPred
            PEst = PPred

        # store data history
        hxTrue.append(xTrue); hxOdom.append(xOdom); hxEst.append(xEst); hxP.append(PEst)

    hxTrue, hxOdom, hxEst = np.hstack(hxTrue), np.hstack(hxOdom), np.hstack(hxEst)
    hxP = np.array(hxP)
    hxError = hxEst - hxTrue
    hxError[2, :] = (hxError[2, :] + pi) % (2 * pi) - pi
    hxVar = np.sqrt(np.einsum("kii->ik", hxP))       # écarts-types (3 x nSteps)
    return dict(Map=Map, hxTrue=hxTrue, hxOdom=hxOdom, hxEst=hxEst, hxP=hxP, hxError=hxError, hxVar=hxVar, with_failure=with_failure)


def compute_metrics(res, burn=200):
    """
    RMSE, part du temps dans +-3 sigma et NEES moyen, après la phase de convergence
    """
    e, s = res["hxError"][:, burn:], res["hxVar"][:, burn:]
    nees = np.einsum("ik,kij,jk->k", e, np.linalg.inv(res["hxP"][burn:]), e)
    return dict(rmse_pos=np.sqrt((e[:2] ** 2).sum(0).mean()), rmse_theta=np.rad2deg(np.sqrt((e[2] ** 2).mean())), in3sigma=100 * np.mean(np.all(np.abs(e) <= 3 * s, axis=0)), nees=nees.mean())


def plot_simulation(res, title, filename, failure_start=2500, failure_end=3500):
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])
    hxTrue, hxOdom, hxEst, hxP = res["hxTrue"], res["hxOdom"], res["hxEst"], res["hxP"]
    hxError, hxVar = res["hxError"], res["hxVar"]

    # Plot true landmark and trajectory
    ax1.plot(res["Map"][0, :], res["Map"][1, :], "*k")
    ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")

    # Plot odometry trajectory
    ax1.plot(hxOdom[0, :], hxOdom[1, :], "-g", label="Odom")

    # Plot estimated trajectory an pose covariance (une ellipse tous les 500 pas)
    ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF")
    # plot_covariance_ellipse(hxEst[:, 0:1], hxP[0], ax1, "--r")
    # plot_covariance_ellipse(hxEst[:, nSteps-1:nSteps], hxP[nSteps-1], ax1, "--r")
    # for k in list(range(0, nSteps, 1000)) + [nSteps - 1]:
        # plot_covariance_ellipse(hxEst[:, k:k+1], hxP[k], ax1, "--r")
    if res["with_failure"]:
        ax1.plot(hxEst[0, failure_start:failure_end], hxEst[1, failure_start:failure_end], "-", color="orange", lw=2, label="EKF (panne)")

    ax1.axis([-70, 70, -70, 70])
    ax1.grid(True)
    ax1.legend()

    # plot errors curves
    for ax, i, label in [(ax3, 0, 'x'), (ax4, 1, 'y'), (ax5, 2, r"$\theta$")]:
        ax.plot(hxError[i, :], 'b', lw=0.6)
        ax.plot(3.0 * hxVar[i, :], 'r')
        ax.plot(-3.0 * hxVar[i, :], 'r')
        if res["with_failure"]:
            ax.axvline(failure_start, color="black", lw=1.5)
            ax.axvline(failure_end, color="black", lw=1.5)
        if label in ["x", "y"]:
            lim = 10
        else:
            lim = max(4 * np.percentile(hxVar[i, 200:], 95), 1.5 * np.percentile(np.abs(hxError[i, 200:]), 99.5))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(label)
    m = compute_metrics(res)
    ax3.set_title(f"Real error (blue) and 3$\\sigma$ covariances (red) - RMSE pos {m['rmse_pos']:.2f} m, θ {m['rmse_theta']:.2f}°, NEES {m['nees']:.2f}", fontsize=10)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=110)
    plt.close(fig)


def run_and_report(name, title, filename=None, **kwargs):
    res = run_simulation(**kwargs)
    m = compute_metrics(res)
    print(f"{name:<48} | {m['rmse_pos']:5.2f} m | {m['rmse_theta']:6.2f}° | {m['in3sigma']:5.1f} % | {m['nees']:7.2f}")
    if filename is not None:
        plot_simulation(res, title, filename, failure_start=kwargs.get("failure_start", 2500), failure_end=kwargs.get("failure_end", 3500))
    return res, m


if __name__ == "__main__":
    print(f"{'scénario':<48} | RMSE pos | RMSE θ  | ±3σ     | NEES (att. 3)")

    # ---- Q1 : réglage du draft, puis réglage exact
    print("--- Q1 : normal")
    run_and_report("draft (10 ; 10)", "Q1 - EKF, réglage du draft (QEst = 10 QTrue, PYEst = 10 PYTrue)", "figures/EKFLocalization_Q1.png")
    run_and_report("exact (1 ; 1)", "Q1 - EKF réglé sur les vrais bruits", "figures/EKFLocalization_Q1_exact.png", q_est=(1, 1, 1), py_est=(1, 1))

    # ---- Q2 : panne capteur 2500 < k < 3500
    failure_start = 2500
    failure_end = 3500
    print(f"--- Q2 : panne entre {failure_start} et {failure_end}")
    run_and_report("panne, draft (10 ; 10)", f"Q2 - panne capteur {failure_start} < k < {failure_end} (réglage du draft)", f"figures/EKFLocalization_Q2_{failure_start}_{failure_end}.png", with_failure=True, failure_start=failure_start, failure_end=failure_end)
    run_and_report("panne, exact (1 ; 1)", f"Q2 - panne capteur {failure_start} < k < {failure_end} (réglage exact)", f"figures/EKFLocalization_Q2_exact_{failure_start}_{failure_end}.png", with_failure=True, q_est=(1, 1, 1), py_est=(1, 1), failure_start=failure_start, failure_end=failure_end)

    failure_start = 1000
    failure_end = 2500
    print(f"--- Q2 : panne entre {failure_start} et {failure_end}")
    run_and_report("panne, draft (10 ; 10)", f"Q2 - panne capteur {failure_start} < k < {failure_end} (réglage du draft)", f"figures/EKFLocalization_Q2_{failure_start}_{failure_end}.png", with_failure=True, failure_start=failure_start, failure_end=failure_end)
    run_and_report("panne, exact (1 ; 1)", f"Q2 - panne capteur {failure_start} < k < {failure_end} (réglage exact)", f"figures/EKFLocalization_Q2_exact_{failure_start}_{failure_end}.png", with_failure=True, q_est=(1, 1, 1), py_est=(1, 1), failure_start=failure_start, failure_end=failure_end)

    # ---- Q3 : erreurs de modèle (facteurs sur QTrue ; PYTrue)
    print("--- Q3 : erreurs de modèle")
    SCENARIOS = [
        # nom, q_est, py_est, figure ?
        ("exact (1 ; 1)",                                 (1, 1, 1),     (1, 1),       None),
        ("draft (10 ; 10)",                               (10, 10, 10),  (10, 10),     None),
        ("sur-estimés (100 ; 100)",                       (100,) * 3,    (100, 100),   "3_sur_estimes"),
        ("sous-estimés (0.01 ; 0.01)",                    (0.01,) * 3,   (0.01, 0.01), "3_sous_estimes"),
        ("odom bruitée, perception précise (100 ; 0.01)", (100,) * 3,    (0.01, 0.01), "3_odom_bruitee_percep_precise"),
        ("odom précise, perception bruitée (0.01 ; 100)", (0.01,) * 3,   (100, 100),   "3_odom_precise_percep_bruitee"),
        ("distance x100, direction exacte",               (1, 1, 1),     (100, 1),     "3_distance_surestimee"),
        ("distance x1e4, direction exacte",               (1, 1, 1),     (1e4, 1),     None),
    ]
    for name, q, py, fig_name in SCENARIOS:
        run_and_report(name, f"Q3 - {name}", None if fig_name is None else f"figures/EKFLocalization_Q{fig_name}.png", q_est=q, py_est=py)

    # ---- Q4 : observabilité partielle (bruits du filtre exacts)
    print("--- Q4 : observabilité partielle")
    for partial_obs in [None, "distance", "direction"]:
        run_and_report(f"partial_obs = {partial_obs}", f"Q4 - partial_obs = {partial_obs}", f"figures/EKFLocalization_Q4_{partial_obs}.png", partial_obs=partial_obs, q_est=(1, 1, 1), py_est=(1, 1))
