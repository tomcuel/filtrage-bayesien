import os
import kalman as kalman
import numpy as np
import graph_helper as gh
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

N = 40
SEED = 1                          # même graine pour tous les runs -> même vérité et mêmes mesures
GAP = (10, 20)                    # Q1b : pas de mesure pour 10 < k < 20
FACTORS = [0.01, 0.1, 1, 10, 100] # Q1c/d : facteurs appliqués au Q / R du FILTRE

X_TRUE_0 = np.array([20.,
                     -10.,
                     5.0,
                     0.,
                     5.,
                     0.])

X0 = np.zeros((6,))
P0 = np.zeros((6,6))
P0[0,0] =10000
P0[1,1] =10000
P0[2,2] =10000
P0[3,3] =100
P0[4,4] =100
P0[5,5] =10

DeltaT = 1

Fk = np.eye(6,6)
Fk[0,3] = DeltaT
Fk[1,4] = DeltaT
Fk[2,5] = DeltaT

Qk = np.eye(6,6)
Qk[0,0] = 0.0001
Qk[1,1] = 0.0001
Qk[2,2] = 0.0001
Qk[3,3] = 0.01
Qk[4,4] = 0.01
Qk[5,5] = 0.01

Hk = np.zeros((3,6))
Hk[0,0] = 1
Hk[1,1] = 1
Hk[2,2] = 1

Rk = np.array([
    [100.0, 0.0, 0.0],
    [0.0, 100.0, 0.0],
    [0.0,0.0,100.0]])

# Q(k) et R(k) sont les VRAIS bruits (utilisés par le simulateur).
# Le filtre utilise q_factor * Q(k) et r_factor * R(k).
def F(k):
    return Fk

def Q(k):
    return Qk

def H(k):
    return Hk

def R(k):
    return Rk


XTrue = Y = XEst = PEst = Gain = None

def simulate_x_step(k):
    global XTrue
    XTrue[k+1] = F(k) @ XTrue[k] + np.linalg.cholesky(Q(k)) @ np.random.randn(6)

def simulate_y(k):
    global XTrue, Y
    Y[k+1] = H(k) @ XTrue[k+1] + np.linalg.cholesky(R(k)) @ np.random.randn(3)


def run(q_factor=1.0, r_factor=1.0, is_measure_gap=False):
    """Simule le système et applique le filtre. Remplit les variables globales XTrue, Y, XEst, PEst, Gain."""
    global XTrue, Y, XEst, PEst, Gain
    np.random.seed(SEED)

    XTrue = np.zeros((N + 1, 6))
    XTrue[0] = X_TRUE_0
    Y = np.full((N + 1, 3), np.nan)
    XEst = np.zeros((N + 1, 6))
    PEst = np.zeros((N + 1, 6, 6))
    Gain = np.full((N + 1, 6, 3), np.nan)      # NaN quand il n'y a pas de correction
    XEst[0] = X0
    PEst[0] = P0

    kf = kalman.KalmanFilter(X0, P0)

    for k in range(N):
        # simulation : toujours faite, pour garder la même suite aléatoire d'un run à l'autre
        simulate_x_step(k)
        simulate_y(k)

        # filtre
        kf.predict(F(k), q_factor * Q(k))
        if not (is_measure_gap and GAP[0] < k < GAP[1]):
            innovation, S, K = kf.update(Y[k+1], H(k+1), r_factor * R(k+1))
            Gain[k+1] = K
        else:
            Y[k+1] = np.nan                        # mesure non disponible (pour l'affichage)

        XEst[k+1] = kf.x
        PEst[k+1] = kf.P


def nees():
    e = XEst - XTrue
    return np.einsum("ki,kij,kj->k", e, np.linalg.inv(PEst), e)


LABELS = ["x", "y", "z", "vx", "vy", "vz"]

def plot_estimation(title, filename, is_measure_gap=False):
    """
    Q1a / Q1b : vérité, estimation et bande 3 sigma pour les 6 composantes
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    t = np.arange(N+1)
    for i in range(6):
        ax = axes[i // 3, i % 3]
        sigma = PEst[:, i, i] ** 0.5
        gh.plot_with_interval(
            t,
            curves=[
                {"y": XTrue[:, i], "label": f"{LABELS[i]} réelle", "color": "blue"},
                {"y": XEst[:, i],  "label": f"{LABELS[i]} estimée", "color": "red"}
            ],
            lower=XEst[:, i] - 3 * sigma,
            upper=XEst[:, i] + 3 * sigma,
            interval_label=r"incertitude ($3\sigma$)",
            xlabel="temps (s)" if i >= 3 else None,
            ylabel="position (m)" if i < 3 else "vitesse (m/s)",
            ax=ax
        )
        if i < 3:
            ax.plot(t, Y[:, i], "g.", ms=4, label="mesure")
            ax.set_ylim(XTrue[:, i].min() - 60, XTrue[:, i].max() + 60)
        else:
            ax.set_ylim(XTrue[:, i].mean() - 12, XTrue[:, i].mean() + 12)
        if is_measure_gap:
            ax.axvline(GAP[0] + 1, color="black", linestyle="--", alpha=.8)
            ax.axvline(GAP[1], color="black", linestyle="--", alpha=.8)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)


def sweep(param, is_measure_gap=False):
    """
    Q1c (param="Q"), Q1d (param="R"), Q et R ensemble (param="QR").
    Superpose les estimations (x et vx) pour chaque facteur, avec une couleur par facteur.
    Retourne les gains au cours du temps pour la figure de synthèse.
    """
    colors = plt.cm.viridis(np.linspace(0, .9, len(FACTORS)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    t = np.arange(N+1)
    gains = {}
    print(f"\n--- variation de {param} ---")
    print(" facteur | RMSE pos k>=25 | RMSE vit k>=25 | NEES moyen k>=10 (att. 6) | K_xx final")

    for factor, c in zip(FACTORS, colors):
        q_factor = factor if param in ("Q", "QR") else 1
        r_factor = factor if param in ("R", "QR") else 1
        run(q_factor, r_factor)
        gains[factor] = Gain.copy()

        for i in range(6):
            ax = axes[i // 3, i % 3]
            ax.plot(t, XEst[:, i], color=c, linestyle="-", linewidth=1.5, label=f"{param}x{factor}")

            sigma = PEst[:, i, i] ** 0.5
            ax.fill_between(t, XEst[:, i] - 3 * sigma, XEst[:, i] + 3 * sigma, color=c, alpha=0.2)
            
            if i < 3:
                ax.set_ylim(XTrue[:, i].min() - 60, XTrue[:, i].max() + 60)
            else:
                ax.set_ylim(XTrue[:, i].mean() - 12, XTrue[:, i].mean() + 12)
            if is_measure_gap:
                ax.axvline(GAP[0] + 1, color="black", linestyle="--", alpha=.8)
                ax.axvline(GAP[1], color="black", linestyle="--", alpha=.8)

            ax.set_xlabel("temps (s)" if i >= 3 else None)
            ax.set_ylabel("position (m)" if i < 3 else "vitesse (m/s)")
            ax.set_title(f"{LABELS[i]} estimé et bande $3\\sigma$")
        
        e = XEst - XTrue
        print(f" {factor:7} | {np.sqrt((e[25:, :3]**2).mean()):14.2f} | {np.sqrt((e[25:, 3:]**2).mean()):14.3f} | {nees()[10:].mean():25.2f} | {Gain[-1, 0, 0]:10.3f}")

    axes[0, 0].plot(t, Y[:, 0], "g.", ms=4, label="mesure")
    axes[0, 1].plot(t, Y[:, 1], "g.", ms=4, label="mesure")
    axes[0, 2].plot(t, Y[:, 2], "g.", ms=4, label="mesure")

    for ax in axes.ravel():
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f"figures/tp1_part1_3_{param}.png", dpi=120)
    return gains



def plot_gains(all_gains):
    """
    Gain de Kalman au cours du temps, et gain final, selon la variation de Q et de R
    """
    colors = plt.cm.viridis(np.linspace(0, .9, len(FACTORS)))
    params = list(all_gains.keys())
    print(len(params), "paramètres : ", params)
    fig, axes = plt.subplots(len(params), 3, figsize=(16, 3.8 * len(params)))
    axes = np.atleast_2d(axes)
    t = np.arange(N+1)
    for row, param in enumerate(params):
        for factor, c in zip(FACTORS, colors):
            K = all_gains[param][factor]
            axes[row, 0].plot(t, K[:, 0, 0], color=c, label=f"{param}x{factor}")
            axes[row, 1].plot(t, K[:, 3, 0], color=c, label=f"{param}x{factor}")
        final_pos = [all_gains[param][f][-1, 0, 0] for f in FACTORS]
        final_vel = [all_gains[param][f][-1, 3, 0] for f in FACTORS]
        axes[row, 2].semilogx(FACTORS, final_pos, "o-", label=r"$K_{x,x}$ final (position)")
        axes[row, 2].semilogx(FACTORS, final_vel, "s-", label=r"$K_{v_x,x}$ final (vitesse)")
        axes[row, 0].set_title(f"variation de {param} : gain position $K_{{x,x}}$")
        axes[row, 1].set_title(f"variation de {param} : gain vitesse $K_{{v_x,x}}$")
        axes[row, 2].set_title(f"gain final en fonction du facteur sur {param}")
        axes[row, 0].set_xlabel("temps (s)"); axes[row, 1].set_xlabel("temps (s)")
        axes[row, 2].set_xlabel(f"facteur sur {param}")
        for ax in axes[row]:
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/tp1_part1_3_gains_x.png")

    fig, axes = plt.subplots(len(params), 3, figsize=(16, 3.8 * len(params)))
    axes = np.atleast_2d(axes)
    t = np.arange(N+1)
    for row, param in enumerate(params):
        for factor, c in zip(FACTORS, colors):
            K = all_gains[param][factor]
            axes[row, 0].plot(t, K[:, 1, 1], color=c, label=f"{param}x{factor}")
            axes[row, 1].plot(t, K[:, 4, 1], color=c, label=f"{param}{factor}")
        final_pos = [all_gains[param][f][-1, 1, 1] for f in FACTORS]
        final_vel = [all_gains[param][f][-1, 4, 1] for f in FACTORS]
        axes[row, 2].semilogx(FACTORS, final_pos, "o-", label=r"$K_{y,y}$ final (position)")
        axes[row, 2].semilogx(FACTORS, final_vel, "s-", label=r"$K_{v_y,y}$ final (vitesse)")
        axes[row, 0].set_title(f"variation de {param} : gain position $K_{{y,y}}$")
        axes[row, 1].set_title(f"variation de {param} : gain vitesse $K_{{v_y,y}}$")
        axes[row, 2].set_title(f"gain final en fonction du facteur sur {param}")
        axes[row, 0].set_xlabel("temps (s)"); axes[row, 1].set_xlabel("temps (s)")
        axes[row, 2].set_xlabel(f"facteur sur {param}")
        for ax in axes[row]:
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/tp1_part1_3_gains_y.png")

    fig, axes = plt.subplots(len(params), 3, figsize=(16, 3.8 * len(params)))
    axes = np.atleast_2d(axes)
    t = np.arange(N+1)
    for row, param in enumerate(params):
        for factor, c in zip(FACTORS, colors):
            K = all_gains[param][factor]
            axes[row, 0].plot(t, K[:, 2, 2], color=c, label=f"{param}x{factor}")
            axes[row, 1].plot(t, K[:, 5, 2], color=c, label=f"{param}x{factor}")
        final_pos = [all_gains[param][f][-1, 2, 2] for f in FACTORS]
        final_vel = [all_gains[param][f][-1, 5, 2] for f in FACTORS]
        axes[row, 2].semilogx(FACTORS, final_pos, "o-", label=r"$K_{z,z}$ final (position)")
        axes[row, 2].semilogx(FACTORS, final_vel, "s-", label=r"$K_{v_z,z}$ final (vitesse)")
        axes[row, 0].set_title(f"variation de {param} : gain position $K_{{z,z}}$")
        axes[row, 1].set_title(f"variation de {param} : gain vitesse $K_{{v_z,z}}$")
        axes[row, 2].set_title(f"gain final en fonction du facteur sur {param}")
        axes[row, 0].set_xlabel("temps (s)"); axes[row, 1].set_xlabel("temps (s)")
        axes[row, 2].set_xlabel(f"facteur sur {param}")
        for ax in axes[row]:
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/tp1_part1_3_gains_z.png")

if __name__ == "__main__":
    print("XTrue0", X_TRUE_0)
    print("X0", X0)
    print("P0", P0)
    print("Fk", Fk)
    print("Qk", Qk)
    print("Hk", Hk)
    print("Rk", Rk)

    # Q1a : réglage nominal
    run()
    plot_estimation("Q1a - Kalman GNSS, réglage nominal", "figures/tp1_part1_1.png")
    sigma = np.sqrt(np.einsum("kii->ki", PEst))
    print("\nQ1a  sigma final :", np.round(sigma[-1], 3), "| NEES moyen k>=10 (att. 6) :", nees()[10:].mean().round(2))

    # Q1b : trou de mesure
    run(is_measure_gap=True)
    plot_estimation(f"Q1b - trou de mesure pour {GAP[0]} < k < {GAP[1]}", "figures/tp1_part1_2.png", is_measure_gap=True)
    print("Q1b  sigma_x :", {k: round(float(PEst[k, 0, 0] ** 0.5), 2) for k in [10, 11, 15, 20, 21, 25]})

    # Q1c / Q1d (+ Q et R ensemble) : variation des bruits du filtre, et gains
    q_gains = sweep("Q", is_measure_gap=False)
    r_gains = sweep("R", is_measure_gap=False)
    qr_gains = sweep("QR", is_measure_gap=False)
    plot_gains({"Q": q_gains, "R": r_gains, "QR": qr_gains})

    plt.show()
