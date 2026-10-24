"""
Nouvelle version : tous les paramètres à modifier sont dans config
--> évite de lancer pleins de code similaires sans réelle reproductibilité
--> copie colle exactement le brouillon donné, juste met le tout sous forme de fonctions congigurables
  - CONFIG                 : tous les paramètres à modifier
  - run_simulation(cfg)    : la boucle du fichier fourni, renvoie l'historique et des métriques
  - plot_results(res)      : trace trajectoires + erreurs (+ nuages de particules)
  - sweep_configs(configs) : compare plusieurs réglages et affiche un tableau
  - q0() ... q5()          : les études demandées
"""
from math import sin, cos, atan2, pi
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Configuration à modifier pour chaque expérience
# =====================================================================
CONFIG = dict(
    # Filtre
    N_PARTICLES=1000,            # nombre de particules
    INIT_STD=[1.0, 1.0, 0.1],    # écart-type du nuage initial (x, y, theta)

    # Bruit modélisé par le filtre = facteur * bruit vrai (facteur sur la VARIANCE)
    #   < 1 : filtre trop optimiste | = 1 : filtre exact | > 1 : filtre pessimiste
    Q_FACTOR=2.0,                # mouvement  (QEst = Q_FACTOR * QTrue)
    R_FACTOR=2.0,                # perception (PYEst = R_FACTOR * PYTrue)

    # Poids
    WEIGHTS="likelihood",        # "likelihood" : w = p(z|x) (fichier fourni) | "previous" : w = w_prec * p(z|x)
    LIK_FLOOR=1e-3,              # plancher de vraisemblance (fichier fourni)

    # Rééchantillonnage
    RESAMPLING_METHOD="systematic",   # "multinomial", "stratified" ou "systematic" (low variance du fichier fourni)
    RESAMPLING_TRIGGER="always",      # "always" (bootstrap), "neff" (SIR : si Neff < NEFF_RATIO * N), "never" (SIS)
    NEFF_RATIO=0.5,

    # Simulation
    N_STEPS=1000,
    FAILURE_START=None,          # début de la panne capteur (None : pas de panne)
    FAILURE_END=None,            # fin de la panne capteur (exclu)
    RECOVERY_THRESHOLD=1.0,      # erreur (m) sous laquelle on considère le filtre "récupéré"
    SNAPSHOTS=(),                # instants où l'on garde une copie du nuage (figures)

    # Divers
    SEED=0,                      # graine (simulation identique d'un réglage à l'autre)
    VERBOSE=True,
)


# =====================================================================
# Global variables (initialisées dans run_simulation)
# =====================================================================
cfg = dict(CONFIG)
nParticles = nSteps = None
Map = xTrue = xOdom = None
QTrue = np.diag([0.02, 0.02, 1*pi/180]) ** 2      # True covariance of errors used for simulating robot movements
PYTrue = np.diag([0.5, 1*pi/180]) ** 2
rng_world = rng_filt = None                         # deux générateurs : le monde ne dépend pas du réglage du filtre


# =====================================================================
# Helper functions
# =====================================================================
# fit angle between -pi and pi (fonctionne aussi sur un tableau de particules)
def angle_wrap(a):
    return (a + pi) % (2 * pi) - pi


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


# =====================================================================
# Simulator functions
# =====================================================================
# return true control at step k
def get_robot_control(k):
    global nSteps
    # generate  sin trajectory
    u = np.array([[0, 0.1,  0.1*np.pi / 180 * sin(3*np.pi * k / nSteps)]]).T
    return u


# simulate new true robot position
def simulate_world(k):
    global xTrue
    u = get_robot_control(k)
    xTrue = tcomp(xTrue, u)
    xTrue[2, 0] = angle_wrap(xTrue[2, 0])


# computes and returns noisy odometry
def get_odometry(k):
    global xOdom  # internal to robot low-level controller
    global QTrue

    u = get_robot_control(k)
    xnow = tcomp(xOdom, u)
    uNoise = np.sqrt(QTrue) @ rng_world.standard_normal(3)
    uNoise = np.array([uNoise]).T
    xnow = tcomp(xnow, uNoise)
    xOdom = xnow
    u = u + uNoise
    return xnow, u


# generate a noisy observation of a random feature
def get_observation(k):
    global Map, xTrue, PYTrue
    # tirages faits même pendant la panne : le monde reste identique d'un réglage à l'autre
    iFeature = rng_world.integers(0, Map.shape[1])
    zNoise = np.sqrt(PYTrue) @ rng_world.standard_normal(2)
    zNoise = np.array([zNoise]).T
    if cfg["FAILURE_START"] is not None and cfg["FAILURE_START"] <= k < cfg["FAILURE_END"]:
        z = None                 # sensor failure
        iFeature = -1
    else:
        z = observation_model(xTrue, iFeature, Map) + zNoise
        z[1, 0] = angle_wrap(z[1, 0])
    return [z, iFeature]


# =====================================================================
# model functions (x peut contenir toutes les particules : 3 x N)
# =====================================================================
# evolution model (f)
def motion_model(x, u):
    c, s = np.cos(x[2]), np.sin(x[2])
    xPred = np.empty_like(x)
    xPred[0] = x[0] + c * u[0] - s * u[1]
    xPred[1] = x[1] + s * u[0] + c * u[1]
    xPred[2] = angle_wrap(x[2] + u[2])
    return xPred


# observation model (h)
def observation_model(xVeh, iFeature, Map):
    Delta = Map[0:2, iFeature:(iFeature+1)] - xVeh[0:2]
    z = np.vstack((np.hypot(Delta[0], Delta[1]), np.arctan2(Delta[1], Delta[0]) - xVeh[2]))
    z[1] = angle_wrap(z[1])
    return z


# =====================================================================
# Particle filter implementation
# =====================================================================
# Particle filter resampling
def re_sampling(px, pw):
    """
    multinomial : N tirages indépendants dans la loi des poids
    stratified  : un tirage uniforme dans chaque intervalle [i/N, (i+1)/N[
    systematic  : un seul tirage, décalé de 1/N (low variance du fichier fourni)
    """
    w_cum = np.cumsum(pw / np.sum(pw))
    w_cum[-1] = 1.0
    base = np.arange(0.0, 1.0, 1 / nParticles)
    if cfg["RESAMPLING_METHOD"] == "multinomial":
        re_sample_id = np.sort(rng_filt.uniform(0, 1, nParticles))
    elif cfg["RESAMPLING_METHOD"] == "stratified":
        re_sample_id = base + rng_filt.uniform(0, 1 / nParticles, nParticles)
    else:
        re_sample_id = base + rng_filt.uniform(0, 1 / nParticles)
    indexes = np.searchsorted(w_cum, re_sample_id)

    px = px[:, indexes]
    pw = np.ones(nParticles) / nParticles    # après tirage, toutes les particules se valent

    return px, pw, indexes


# weighted mean (moyenne circulaire pour le cap) and std of the particles
def estimate(xParticles, L):
    w = L / np.sum(L)
    xEst = np.array([[xParticles[0] @ w], [xParticles[1] @ w], [atan2(np.sin(xParticles[2]) @ w, np.cos(xParticles[2]) @ w)]])
    d = xParticles - xEst
    d[2] = angle_wrap(d[2])
    xSTD = np.sqrt((d ** 2) @ w).reshape(3, 1)
    return xEst, xSTD


# =====================================================================
# Simulation
# =====================================================================
def run_simulation(user_cfg=None):
    global cfg, nParticles, nSteps, Map, xTrue, xOdom, rng_world, rng_filt
    cfg = dict(CONFIG)
    if user_cfg:
        cfg.update(user_cfg)
    rng_world = np.random.default_rng(cfg["SEED"])
    rng_filt = np.random.default_rng(cfg["SEED"] + 1000)

    # Nb of particle in the filter
    nParticles = cfg["N_PARTICLES"]

    # Simulation length
    nSteps = cfg["N_STEPS"]

    # Location of landmarks
    Map = 120*rng_world.random((2, 20))-60

    # Modeled errors used in the particle filter
    QEst = cfg["Q_FACTOR"] * QTrue
    PYEst = cfg["R_FACTOR"] * PYTrue

    # initial conditions
    xTrue = np.array([[1, -50, 0]], dtype=float).T
    xOdom = xTrue

    # initial conditions: - a point cloud around truth
    xParticles = xTrue + np.diag(cfg["INIT_STD"]) @ rng_filt.standard_normal((3, nParticles))

    # initial weights
    L = np.ones((nParticles))/nParticles

    # initial estimate
    xEst, xSTD = estimate(xParticles, L)

    # Init history matrixes (listes, empilées à la fin)
    hxEst, hxTrue, hxOdom, hxSTD = [xEst], [xTrue], [xOdom], [xSTD]
    err = xEst - xTrue
    err[2, 0] = angle_wrap(err[2, 0])
    hxError = [err]
    hxNeff, hxResamp, distinct, snaps = [nParticles], [False], [], {}
    n_resamples = 0

    t0 = time.time()
    for k in range(1, nSteps):
        # Simlulate robot motion
        simulate_world(k)

        # Get odometry measurements
        xOdom, u = get_odometry(k)

        # do prediction
        # for each particle we add control vector AND noise (toutes les particules d'un coup)
        xParticles = motion_model(xParticles, u + np.sqrt(QEst) @ rng_filt.standard_normal((3, nParticles)))

        # observe a random feature
        [z, iFeature] = get_observation(k)

        if z is not None:
            # Predict observation from the particles positions
            zPred = observation_model(xParticles, iFeature, Map)

            # Innovation : perception error
            Innov = z - zPred
            Innov[1] = angle_wrap(Innov[1])

            # Compute particle weight using gaussian model
            lik = np.exp(-0.5 * np.sum(Innov * (np.linalg.inv(PYEst) @ Innov), axis=0)) + cfg["LIK_FLOOR"]
            if cfg["WEIGHTS"] == "previous":
                L = L * lik              # w_k = w_{k-1} * p(z_k | x_k) : échantillonnage d'importance séquentiel
            else:
                L = lik                  # w_k = p(z_k | x_k) : vraisemblance seule (fichier fourni)
            L = L / np.sum(L)

        # Compute position as weighted mean of particles, and particles std deviation
        xEst, xSTD = estimate(xParticles, L)
        neff = 1.0 / np.sum(L ** 2)
        if k in cfg["SNAPSHOTS"]:
            snaps[k] = xParticles.copy()

        # Particle resampling : bootstrap (always), SIR (neff) ou SIS (never) ; inutile pendant une panne
        trigger = cfg["RESAMPLING_TRIGGER"]
        do_resample = trigger == "always" or (trigger == "neff" and neff < cfg["NEFF_RATIO"] * nParticles)
        if do_resample and z is not None:
            xParticles, L, indexes = re_sampling(xParticles, L)
            n_resamples += 1
            distinct.append(len(np.unique(indexes)) / nParticles)     # part de particules différentes conservées

        # store data history
        hxTrue.append(xTrue); hxOdom.append(xOdom); hxEst.append(xEst); hxSTD.append(xSTD)
        err = xEst - xTrue
        err[2, 0] = angle_wrap(err[2, 0])
        hxError.append(err)
        hxNeff.append(neff); hxResamp.append(bool(do_resample and z is not None))

    elapsed = time.time() - t0
    hxTrue, hxOdom, hxEst, hxError, hxSTD = (np.hstack(h) for h in (hxTrue, hxOdom, hxEst, hxError, hxSTD))
    tErrors = np.sqrt(np.square(hxError[0, :]) + np.square(hxError[1, :]))

    # Métriques
    res = dict(cfg=cfg, Map=Map, hTrue=hxTrue, hOdom=hxOdom, hEst=hxEst, hErr=hxError, hSTD=hxSTD, tErr=tErrors, hNeff=np.array(hxNeff), resampled=np.array(hxResamp), xP=xParticles, L=L, snaps=snaps)
    T, N = nSteps, nParticles
    burn = 50                                                    # convergence initiale exclue
    ok = np.ones(T, bool); ok[:burn] = False
    if cfg["FAILURE_START"] is not None:                         # précision "hors panne" : on exclut la panne
        ok[cfg["FAILURE_START"]:cfg["FAILURE_END"] + 50] = False
    res["time_s"] = elapsed
    res["n_resamples"] = n_resamples
    res["mean_err"] = float(np.mean(tErrors[ok]))
    res["theta_err"] = float(np.degrees(np.mean(np.abs(hxError[2, ok]))))
    res["in3sigma"] = float(100 * np.mean(np.all(np.abs(hxError[:, burn:]) <= 3 * hxSTD[:, burn:], axis=0)))
    res["neff"] = float(np.mean(res["hNeff"][burn:]) / N)
    res["distinct"] = float(np.mean(distinct)) if distinct else 1.0
    res["final_err"] = float(np.mean(tErrors[-50:]))
    res["diverged"] = bool(res["final_err"] > 5.0)

    if cfg["FAILURE_START"] is not None and cfg["FAILURE_END"] > cfg["FAILURE_START"]:
        fs, fe = cfg["FAILURE_START"], min(cfg["FAILURE_END"], T)
        res["max_err_during_failure"] = float(np.max(tErrors[fs:fe]))
        after = np.where(tErrors[fe:] < cfg["RECOVERY_THRESHOLD"])[0]
        res["recovery_steps"] = int(after[0]) if len(after) else None  # pas après la fin de la panne

    if cfg["VERBOSE"]:
        print("Mean (var) translation error : {:e} ({:e})".format(np.mean(tErrors), np.var(tErrors)))
    return res


# =====================================================================
# Affichage
# =====================================================================
def plot_results(res, save="ParticleFilter_Localization.png", title=None):
    cfg = res["cfg"]
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(res["Map"][0], res["Map"][1], "*k")
    ax1.plot(res["hTrue"][0], res["hTrue"][1], "-k", label="True")
    ax1.plot(res["hOdom"][0], res["hOdom"][1], "-g", label="Odom")
    ax1.plot(res["hEst"][0], res["hEst"][1], "-r", label="Part. Filt.")
    for i, snap in enumerate(list(res["snaps"].values()) + [res["xP"]]):
        xP = snap[0] if isinstance(snap, tuple) else snap          # snapshot = (particules, poids) ou particules seules
        ax1.scatter(xP[0, :400], xP[1, :400], s=3, alpha=0.4, color="tab:blue", label="particules" if i == 0 else None)
    ax1.axis([-60, 60, -60, 60]); ax1.grid(True); ax1.legend(loc="upper left")

    for i, (lab, row) in enumerate(zip(["x", "y", r"$\theta$"], range(3))):
        ax = fig.add_subplot(3, 2, 2 * (i + 1))
        ax.plot(res["hErr"][row], "b", lw=0.8)
        ax.plot(3 * res["hSTD"][row], "r")
        ax.plot(-3 * res["hSTD"][row], "r")
        if cfg["FAILURE_START"] is not None:
            ax.axvspan(cfg["FAILURE_START"], cfg["FAILURE_END"], color="gray", alpha=0.2)
        e = np.abs(res["hErr"][row, 50:])
        lim = min(8 if row < 2 else 0.6, max(1.0 if row < 2 else 0.15, 2 * np.percentile(e, 99)))   # échelle adaptée
        ax.set_ylim(-lim, lim)
        ax.grid(True); ax.set_ylabel(lab)
        if i == 0:
            ax.set_title(f"Erreur réelle (bleu) et ±3σ des particules (rouge) - err. moy. {res['mean_err']:.2f} m, ±3σ {res['in3sigma']:.0f} %", fontsize=10)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save, dpi=110)
    plt.close(fig)


# =====================================================================
# Outils de balayage / comparaison (utilisés par Q0..Q5)
# =====================================================================
OUTPUT_DIR = "figures"   # dossier où sont sauvegardés CSV et figures
N_SEEDS = 5              # nombre de graines moyennées dans chaque balayage
SWEEP_BASE = {}          # overrides communs à toutes les questions
SHOW_PLOTS = False       # afficher les figures à la fin


def _save_path(fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, fname)


def sweep_configs(configs, title, n_seeds=None, base=None, save_csv=None):
    """
    configs : liste de (label, dict d'overrides de CONFIG)
    Moyenne sur n_seeds graines, affiche un tableau, sauvegarde un CSV, renvoie les lignes
    """
    n_seeds = n_seeds or N_SEEDS
    base = {**SWEEP_BASE, **(base or {})}
    print(f"\n=== {title} ===")
    print(f"{'config':>34} | {'err moy':>8} | {'θ (°)':>6} | {'±3σ':>6} | {'Neff/N':>6} | {'distinct':>8} | "
          f"{'max panne':>9} | {'récup':>6} | {'resamples':>9} | {'diverge':>7} | {'temps':>6}")
    rows = []
    for label, ov in configs:
        rs = [run_simulation({**base, **ov, "SEED": s, "VERBOSE": False}) for s in range(n_seeds)]
        rec = [r["recovery_steps"] for r in rs if r.get("recovery_steps") is not None]
        ok = [r for r in rs if not r["diverged"]] or rs           # précision : runs non divergés
        row = dict(
            config=label,
            mean_err=np.mean([r["mean_err"] for r in ok]),
            theta_err=np.mean([r["theta_err"] for r in ok]),
            in3sigma=np.mean([r["in3sigma"] for r in rs]),
            neff=np.mean([r["neff"] for r in rs]),
            distinct=np.mean([r["distinct"] for r in rs]),
            max_err_failure=np.mean([r.get("max_err_during_failure", np.nan) for r in rs]),
            recovery_steps=np.mean(rec) if rec else float("nan"),
            n_resamples=np.mean([r["n_resamples"] for r in rs]),
            n_diverged=sum(r["diverged"] for r in rs),
            time_s=np.mean([r["time_s"] for r in rs]),
        )
        rows.append(row)
        print(f"{label:>34} | {row['mean_err']:8.3f} | {row['theta_err']:6.2f} | {row['in3sigma']:5.1f}% | "
              f"{row['neff']:6.2f} | {row['distinct']:8.2f} | {row['max_err_failure']:9.2f} | "
              f"{row['recovery_steps']:6.1f} | {row['n_resamples']:9.0f} | {row['n_diverged']}/{n_seeds:<5} | "
              f"{row['time_s']:6.2f}", flush=True)
    if save_csv:
        with open(_save_path(save_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def compare_curves(configs, title, fname, seed=0, base=None):
    """
    Trace l'erreur de translation au cours du temps pour chaque config (une graine)
    """
    base = {**SWEEP_BASE, **(base or {})}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fs = fe = None
    for label, ov in configs:
        r = run_simulation({**base, **ov, "SEED": seed, "VERBOSE": False})
        ax.plot(r["tErr"], label=label, lw=1)
        fs, fe = r["cfg"]["FAILURE_START"], r["cfg"]["FAILURE_END"]
    if fs is not None and fe > fs:
        ax.axvspan(fs, fe, color="gray", alpha=0.2, label="panne capteur")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("pas k"); ax.set_ylabel("erreur de translation (m, échelle log)")
    ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(_save_path(fname), dpi=110)
    plt.close(fig)


def plot_neff(configs, title, fname, seed=0, base=None):
    """
    Pour chaque config : Neff/N au cours du temps (avant rééchantillonnage),
    seuil NEFF_RATIO (si critère neff) et instants de rééchantillonnage (traits rouges en bas)
    """
    base = {**SWEEP_BASE, **(base or {})}
    n = len(configs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n + 1), sharex=True, squeeze=False)
    for ax, (label, ov) in zip(axes[:, 0], configs):
        r = run_simulation({**base, **ov, "SEED": seed, "VERBOSE": False})
        c = r["cfg"]
        ax.plot(r["hNeff"] / c["N_PARTICLES"], lw=1, label="Neff / N")
        steps = np.where(r["resampled"])[0]
        ax.vlines(steps, 0, 0.07, color="red", lw=0.6, label="rééchantillonnage")
        if c["RESAMPLING_TRIGGER"] == "neff":
            ax.axhline(c["NEFF_RATIO"], color="k", ls="--", lw=0.8, label="seuil")
        fs, fe = c["FAILURE_START"], c["FAILURE_END"]
        if fs is not None and fe > fs:
            ax.axvspan(fs, fe, color="gray", alpha=0.2, label="panne capteur")
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
        ax.set_title(f"{label} - {len(steps)} rééchantillonnages", fontsize=9)
        ax.set_ylabel("Neff / N")
    axes[0, 0].legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1, 0].set_xlabel("pas k")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(_save_path(fname), dpi=110)
    plt.close(fig)



def get_snapshot(res, k):
    """
    Nuage (3 x N) et poids à l'instant k, quel que soit le format enregistré dans res["snaps"]
    """
    snap = res["snaps"].get(k)
    if snap is None:                       # instant non enregistré : on prend le nuage final
        print(f"attention : pas de nuage enregistré à k={k} (ajouter k à SNAPSHOTS), nuage final utilisé")
        return res["xP"], np.ones(res["xP"].shape[1])
    if isinstance(snap, tuple):            # (particules, poids)
        return snap[0], snap[1]
    return snap, np.ones(snap.shape[1])    # particules seules : poids égaux


def plot_runs(configs, title, fname, k_snap=500, zoom=None, seed=0, base=None):
    """
    Une ligne par config : trajectoires et nuage à l'instant k_snap (taille des points = poids),
    zoom sur le nuage autour de la vraie pose (demi-largeur zoom, en m), erreur de position et 3σ du nuage
    """
    base = {**SWEEP_BASE, **(base or {})}
    n, ncol = len(configs), 3 if zoom else 2
    fig, axes = plt.subplots(n, ncol, figsize=(5 * ncol, 3.6 * n), squeeze=False, gridspec_kw={"width_ratios": [1] * (ncol - 1) + [1.6]})
    for row, (label, ov) in zip(axes, configs):
        r = run_simulation({**base, **ov, "SEED": seed, "VERBOSE": False, "SNAPSHOTS": (k_snap,)})
        xP, L = get_snapshot(r, k_snap)
        size = 4 + 300 * L / L.max()                                   # gros point = particule lourde
        maps = [(row[0], 60)] + ([(row[1], zoom)] if zoom else [])
        for ax, half in maps:
            ax.plot(r["Map"][0], r["Map"][1], "*k")
            ax.plot(r["hTrue"][0], r["hTrue"][1], "-k", lw=1, label="True")
            ax.plot(r["hOdom"][0], r["hOdom"][1], "-g", lw=1, label="Odom")
            ax.plot(r["hEst"][0], r["hEst"][1], "-r", lw=1, label="Part. Filt.")
            ax.scatter(xP[0], xP[1], s=size if half != 60 else 3, alpha=0.4, color="tab:blue", label=f"particules (k={k_snap})")
            cx, cy = (0, 0) if half == 60 else r["hTrue"][:2, k_snap]
            ax.plot(*r["hTrue"][:2, k_snap], "*", color="red", ms=12)
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        row[0].set_title(label, fontsize=10)
        if zoom:
            row[1].set_title(f"zoom ±{zoom} m autour de la vraie pose (k={k_snap})", fontsize=9)
        ax = row[-1]
        ax.semilogy(r["tErr"], "b", lw=0.8, label="erreur de position")
        ax.semilogy(np.maximum(3 * np.hypot(r["hSTD"][0], r["hSTD"][1]), 1e-3), "r", lw=0.8, label="3σ du nuage")
        ax.set_ylim(1e-3, 200)
        ax.axvline(k_snap, color="k", ls=":", lw=0.8)
        ax.set_title(f"err. moy. {r['mean_err']:.2f} m, ±3σ {r['in3sigma']:.0f} %, Neff/N {r['neff']:.2f}, distinctes {r['distinct']:.2f}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left"); axes[0, -1].legend(fontsize=7, loc="upper right")
    axes[-1, -1].set_xlabel("pas k")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(_save_path(fname), dpi=110)
    plt.close(fig)


# =====================================================================
# Les questions du TP
# =====================================================================
def q0():
    """
    Q0 - Réglage du fichier fourni (référence)
    """
    rows = sweep_configs([("fichier fourni (Q x2, R x2)", {}), ("bruits exacts (Q x1, R x1)", {"Q_FACTOR": 1, "R_FACTOR": 1})],
                         "Q0 - référence", save_csv="Q0_reference.csv")
    plot_results(run_simulation({"VERBOSE": False, "SNAPSHOTS": (0, 250, 500, 750)}), _save_path("Q0_reference.png"),
                 "Réglage du fichier fourni (N = 1000, Q x2, R x2)")
    return rows


def q1():
    """
    Q1 - Influence du bruit de mouvement Q et du bruit de perception R supposés par le filtre
    """
    factors = [0.01, 0.1, 0.5, 1, 2, 10, 100]
    # (a) bruit de mouvement du filtre varie, perception exacte
    cfg_a = [(f"Q x{f} | R x1", {"Q_FACTOR": f, "R_FACTOR": 1.0}) for f in factors]
    rows_a = sweep_configs(cfg_a, "Q1a - bruit de mouvement du filtre", save_csv="Q1a_Qfactor.csv")
    compare_curves(cfg_a[::2], "Q1a - bruit de mouvement du filtre (Q_FACTOR)", "Q1a_Qfactor.png")
    # (b) bruit de perception du filtre varie, mouvement exact
    cfg_b = [(f"Q x1 | R x{f}", {"Q_FACTOR": 1.0, "R_FACTOR": f}) for f in factors]
    rows_b = sweep_configs(cfg_b, "Q1b - bruit de perception du filtre", save_csv="Q1b_Rfactor.csv")
    compare_curves(cfg_b[::2], "Q1b - bruit de perception du filtre (R_FACTOR)", "Q1b_Rfactor.png")
    # (c) combinaisons croisées petit / égal / grand
    cfg_c = [(f"Q x{q} | R x{r}", {"Q_FACTOR": q, "R_FACTOR": r}) for q in [0.1, 1, 10] for r in [0.1, 1, 10]]
    rows_c = sweep_configs(cfg_c, "Q1c - combinaisons (petit / égal / grand)", save_csv="Q1c_grid.csv")
    plot_results(run_simulation({"Q_FACTOR": 0.1, "R_FACTOR": 1, "VERBOSE": False}), _save_path("Q1_Qpetit.png"), "Q x0.1, R x1 : bruit de mouvement sous-estimé")
    plot_summary(factors, rows_a, rows_b)
    return rows_a, rows_b, rows_c


def plot_summary(factors, rows_a, rows_b):
    """
    Q1 : erreur et consistance en fonction du facteur sur Q (R exact) ou sur R (Q exact)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for rows, st, lab in [(rows_a, "o-", "Q_FACTOR (R exact)"), (rows_b, "s-", "R_FACTOR (Q exact)")]:
        ax[0].loglog(factors, [r["mean_err"] for r in rows], st, label=lab)
        ax[1].semilogx(factors, [r["in3sigma"] for r in rows], st, label=lab)
    ax[0].set_title("erreur moyenne de position (m)"); ax[1].set_title("% du temps dans ±3σ")
    for a in ax:
        a.axvline(1, color="k", lw=0.8, ls=":"); a.set_xlabel("facteur sur la variance")
        a.grid(True, which="both", alpha=0.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(_save_path("Q1_resume.png"), dpi=110); plt.close(fig)


def q2():
    """
    Q2 : Influence du nombre de particules
    """
    ns = [10, 30, 100, 300, 1000, 3000, 10000]
    cfgs = [(f"N={n}", {"N_PARTICLES": n}) for n in ns]
    rows = sweep_configs(cfgs, "Q2 : nombre de particules", save_csv="Q2_nparticles.csv")
    compare_curves(cfgs[::2], "Q2 : nombre de particules", "Q2_nparticles.png")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].semilogx(ns, [r["mean_err"] for r in rows], "o-"); ax[0].set_title("erreur moyenne de position (m)")
    ax[1].semilogx(ns, [r["in3sigma"] for r in rows], "o-"); ax[1].set_title("% du temps dans ±3σ")
    ax[2].loglog(ns, [r["time_s"] for r in rows], "o-"); ax[2].set_title("temps de calcul (s, 1000 pas)")
    for a in ax:
        a.set_xlabel("nombre de particules N"); a.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(_save_path("Q2_resume.png"), dpi=110); plt.close(fig)
    return rows


def q3():
    """
    Q3 : Panne capteur courte, longue et très longue : influence du réglage pendant et après la panne
    """
    pannes = {"courte": (400, 500), "longue": (400, 700), "tres_longue": (200, 900)}
    cfgs = []
    for name, (fs, fe) in pannes.items():
        p = {"FAILURE_START": fs, "FAILURE_END": fe}
        cfgs += [(f"{name}, fichier", p),
                 (f"{name}, Q x1 R x1", {**p, "Q_FACTOR": 1, "R_FACTOR": 1}),
                 (f"{name}, Q x10", {**p, "Q_FACTOR": 10}),
                 (f"{name}, N=100", {**p, "N_PARTICLES": 100}),
                 (f"{name}, N=10000", {**p, "N_PARTICLES": 10000})]
    cfgs.append(("longue, previous", {"FAILURE_START": 400, "FAILURE_END": 700, "WEIGHTS": "previous"}))
    rows = sweep_configs(cfgs, "Q3 : panne capteur", save_csv="Q3_panne.csv")
    for name, (fs, fe) in pannes.items():
        plot_results(run_simulation({"FAILURE_START": fs, "FAILURE_END": fe, "SNAPSHOTS": (fs, fe - 1), "VERBOSE": False}),
                     _save_path(f"Q3_panne_{name}.png"), f"Panne capteur ({fs} ≤ k < {fe}), réglage du fichier")
    compare_curves([("N=100", {"N_PARTICLES": 100}), ("N=1000 (fichier)", {}), ("N=10000", {"N_PARTICLES": 10000}),
                    ("N=1000, Q x10", {"Q_FACTOR": 10})], "Q3 : panne très longue (200 ≤ k < 900)",
                   "Q3_tres_longue.png", base={"FAILURE_START": 200, "FAILURE_END": 900})
    return rows


def q4():
    """
    Q4 : Poids (vraisemblance seule / poids précédents) et schéma : bootstrap, SIR, SIS
    """
    schemes = [("bootstrap", {"RESAMPLING_TRIGGER": "always"}),
               ("SIR (Neff < N/2)", {"RESAMPLING_TRIGGER": "neff"}),
               ("SIS (jamais)", {"RESAMPLING_TRIGGER": "never"})]
    cfgs = [(f"{w}, {name}", {**ov, "WEIGHTS": w}) for w in ["likelihood", "previous"] for name, ov in schemes]
    rows = sweep_configs(cfgs, "Q4 : poids et schéma de rééchantillonnage", save_csv="Q4_schemas.csv")
    compare_curves(cfgs, "Q4 : poids et schéma de rééchantillonnage", "Q4_schemas.png")
    plot_neff([(f"previous, {name}", {**ov, "WEIGHTS": "previous"}) for name, ov in schemes] +
              [("likelihood, SIR (Neff < N/2)", {"RESAMPLING_TRIGGER": "neff", "WEIGHTS": "likelihood"})],
              "Q4 : Neff/N et instants de rééchantillonnage", "Q4_neff.png")
    cfgs_ratio = [(f"previous, SIR ratio={r}", {"RESAMPLING_TRIGGER": "neff", "WEIGHTS": "previous", "NEFF_RATIO": r})
                  for r in [0.1, 0.3, 0.5, 0.8]]
    rows_ratio = sweep_configs(cfgs_ratio, "Q4b : seuil NEFF_RATIO", save_csv="Q4b_neff_ratio.csv")
    # trajectoires et nuages : bootstrap, puis les deux SIS
    plot_runs([("poids précédents, bootstrap", {"WEIGHTS": "previous"}),
               ("poids précédents, SIS (jamais)", {"WEIGHTS": "previous", "RESAMPLING_TRIGGER": "never"}),
               ("vraisemblance seule, SIS (jamais)", {"WEIGHTS": "likelihood", "RESAMPLING_TRIGGER": "never"})],
              "Q4 : trajectoires et nuage de particules (taille des points = poids)", "Q4_trajectoires.png", zoom=10)
    return rows, rows_ratio


def q5():
    """
    Q5 : Méthodes de rééchantillonnage : multinomial, stratified, systematic
    """
    out = []
    for n in [1000, 100]:
        cfgs = [(f"N={n}, {m}, {t}", {"N_PARTICLES": n, "RESAMPLING_METHOD": m, "RESAMPLING_TRIGGER": t, "WEIGHTS": "previous"}) 
        for t in ["always", "neff"] for m in ["multinomial", "stratified", "systematic"]]
        out.append(sweep_configs(cfgs, f"Q5 : méthodes de rééchantillonnage (N = {n})", n_seeds=10, save_csv=f"Q5_methodes_N{n}.csv"))
    plot_runs([(f"N=100, {m}, bootstrap", {"N_PARTICLES": 100, "RESAMPLING_METHOD": m, "WEIGHTS": "previous"})
               for m in ["multinomial", "systematic"]],
              "Q5 : N = 100 : multinomial contre systematic (taille des points = poids)", "Q5_trajectoires.png", zoom=1.5)
    return out


QUESTIONS = {"Q0": q0, "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


def run_single():
    """
    Un seul run avec la CONFIG du haut du fichier
    """
    res = run_simulation()
    plot_results(res, save=_save_path("single_run.png"))
    return res


# =====================================================================
# Main function
# =====================================================================
# Valeur utilisée si aucun argument n'est donné : "RUN", "Q0" ... "Q5" ou "ALL"
QUESTION_TO_RUN = "ALL"


def main(question=QUESTION_TO_RUN):
    question = question.upper()
    if question == "RUN":
        run_single()
    elif question == "ALL":
        for name, fn in QUESTIONS.items():
            print(f"\n################  {name}  ################")
            fn()
    elif question in QUESTIONS:
        QUESTIONS[question]()
    else:
        raise SystemExit(f"Question inconnue : {question}. Choix : RUN, ALL, {', '.join(QUESTIONS)}")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TP filtre à particules")
    parser.add_argument("question", nargs="?", default=QUESTION_TO_RUN, help="RUN | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | ALL")
    args, _ = parser.parse_known_args()
    main(args.question)
