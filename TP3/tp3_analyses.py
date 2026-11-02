"""
Q1 : - recherche du cap optimal (q1_manoeuvre.main, pas de 0.1 deg) : q1_grille, q1_trajectoires, q1_incertitudes
     - phi = 0 sans a priori : famille de trajectoires donnant les memes angles, rang de J
     - influence de l'instant de manoeuvre
     - lecture geometrique : composante de la variation de vitesse perpendiculaire a la ligne de visee
Q2 : - experiences A / B / C du TP (Q2b-(1), (2), (3)) : figures et valeurs
     - efficacite (RMSE / borne) et consistance (P EKF / RMSE, NEES) au cours du temps
     - experience A avec un a priori plus serre (cause de l'ecart EKF / BCR)
     - reglage de Q dans l'EKF face a la borne du vrai modele
     - convergence du Monte-Carlo de la borne

Lancement : python tp3_analyses.py [Q1 | Q2 | ALL]
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import q1_manoeuvre as q1
import experiments as ex
from cramer_rao_deterministe import DeterministicDynamicsPCRB


FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------------------------------- Q1
def q1_grille(step_deg=0.1):
    """
    Equivalent de `python q1_manoeuvre.py --step-deg 0.1 --output-dir figures` :
    figures q1_grille.png, q1_trajectoires.png et q1_incertitudes.png dans FIG_DIR
    """
    return q1.main(["--step-deg", str(step_deg), "--output-dir", FIG_DIR, "--output-prefix", "q1"])


def q1_non_observable():
    """
    phi = 0 : l'observateur va tout droit
    Toute trajectoire p_a(k) = o(k) + a (p(k) - o(k)) est rectiligne uniforme et donne exactement les memes angles que la vraie cible
    """
    obs = q1.observer_trajectory(0.0)
    target = np.array([q1.X0_TARGET[:2] + k * q1.DT * q1.X0_TARGET[2:] for k in range(q1.N_MEASURES)])
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    th_err = 0.0
    for a in np.r_[np.linspace(0.1, 0.85, 6), np.linspace(1.25, 3.0, 8)]:   # cibles plus proches (a < 1) ou plus lointaines (a > 1)
        p = obs + a * (target - obs)
        th_err = max(th_err, np.max(np.abs(np.arctan2(p[:, 1] - obs[:, 1], p[:, 0] - obs[:, 0])
                                           - np.arctan2(target[:, 1] - obs[:, 1], target[:, 0] - obs[:, 0]))))
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:red", lw=1.2)
        ax.plot(p[0, 0], p[0, 1], "o", mfc="none", mec="tab:red", ms=5)   # point de depart
    ax.plot(target[:, 0], target[:, 1], "-", color="tab:blue", lw=2.5)
    ax.plot(target[0, 0], target[0, 1], "o", mfc="none", mec="tab:blue", ms=6)
    print(f"phi=0 : ecart d'angle max entre trajectoires de la famille = {th_err:.1e} rad")
    ax.plot(obs[:, 0], obs[:, 1], color="k", lw=2.5, label="observateur (φ = 0)")
    ax.plot([], [], color="tab:blue", lw=2.5, marker="o", mfc="none", label="vraie cible")
    ax.plot([], [], color="tab:red", lw=1.2, marker="o", mfc="none", label="autres cibles, mêmes angles")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="φ = 0 : une infinité de trajectoires donnent les mêmes angles")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.3, ls=":"); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG_DIR + "/q1_non_observable.png", dpi=150); plt.close(fig)

    # rang de J sans a priori (P0 tres grand)
    for phi in (0.0, 114.9):
        obs = q1.observer_trajectory(phi)
        bcr = DeterministicDynamicsPCRB(np.eye(4) * 1e14)
        x = q1.X0_TARGET.copy()
        bcr.update(q1.bearing_jacobian(x, obs[0]), q1.R)
        for k in range(1, q1.N_MEASURES):
            x = q1.F @ x
            bcr.step(q1.F, q1.bearing_jacobian(x, obs[k]), q1.R)
        ev = np.linalg.eigvalsh(bcr.J)
        pos = np.sqrt(np.trace(bcr.bound[:2, :2]))
        print(f"sans a priori, phi={phi:6.1f} : valeurs propres de J = {ev}, BCR position = {pos:.3g} m")


def q1_instant_manoeuvre(instants_detail=(10, 25, 50, 75, 90)):
    """
    Cap optimal et erreur finale selon l'instant de la manoeuvre
    - Gauche : courbes erreur(phi) pour quelques instants 
    - Droite : cap optimal et erreur minimale pour chaque instant
    """
    grid = np.arange(0, 360, 1.0)
    instants = np.arange(5, 100, 5)
    best_phi, best_err, curves = [], [], {}
    for m in instants:
        c = np.sqrt([q1.evaluate_phi(p, manoeuvre_index=m)[0] for p in grid])
        best_phi.append(grid[np.argmin(c)]); best_err.append(c.min())
        if m in instants_detail:
            curves[m] = c
    best_phi, best_err = np.array(best_phi), np.array(best_err)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for m, c in curves.items():
        ax[0].semilogy(grid, c, label=f"manoeuvre à k = {m}")
    ax[0].set(xlabel="cap après manoeuvre φ (degrés)", ylabel="√(BCR_xx + BCR_yy) finale (m)",
              title="Erreur finale selon le cap, pour quelques instants")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
    ax[1].plot(instants, best_phi, "o-", color="tab:blue")
    ax[1].set(xlabel="instant de la manoeuvre k", ylabel="cap optimal φ* (degrés)", title="Cap optimal et erreur minimale")
    ax[1].tick_params(axis="y", colors="tab:blue"); ax[1].yaxis.label.set_color("tab:blue")
    ax2 = ax[1].twinx()
    ax2.semilogy(instants, best_err, "s--", color="tab:red")
    ax2.set_ylabel("erreur finale minimale (m)", color="tab:red"); ax2.tick_params(axis="y", colors="tab:red")
    ax[1].grid(alpha=.3)
    fig.savefig(os.path.join(FIG_DIR, "q1_instant_manoeuvre.png"), dpi=150); plt.close(fig)
    for m, phi, err in zip(instants, best_phi, best_err):
        if m in instants_detail:
            print(f"manoeuvre a k={m:2d} : phi optimal = {phi:5.1f} deg, erreur de position finale = {err:7.1f} m")


def q1_geometrie():
    """
    Variation de vitesse de l'observateur Dv = v_apres - v_avant, et sa composante perpendiculaire
    a la ligne de visee au moment de la manoeuvre
    """
    m = q1.N_MEASURES // 2
    obs = q1.observer_trajectory(0.0)
    tgt = q1.X0_TARGET[:2] + (m - 1) * q1.DT * q1.X0_TARGET[2:]
    los = tgt - obs[m - 1]
    beta = np.degrees(np.arctan2(los[1], los[0]))
    u_perp = np.array([-los[1], los[0]]) / np.linalg.norm(los)
    u_los = los / np.linalg.norm(los)
    grid = np.arange(0, 360, 1.0)
    dv = np.array([[10 * np.cos(np.radians(p)) - 10, 10 * np.sin(np.radians(p))] for p in grid])
    perp = np.abs(dv @ u_perp)
    approche = np.array([10 * np.cos(np.radians(p - beta)) for p in grid])
    print(f"ligne de visee au moment de la manoeuvre : beta = {beta:.1f} deg, distance = {np.linalg.norm(los):.0f} m")
    print(f"|Dv perpendiculaire| maximal pour phi = {grid[np.argmax(perp)]:.0f} deg, nul pour phi = 0 et {grid[1:][np.argmin(perp[1:])]:.0f} deg")
    c = np.array([q1.evaluate_phi(p)[0] for p in grid])
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    ax.plot(grid, np.sqrt(c), color="tab:blue", label="√BCR position finale (m)")
    ax2 = ax.twinx()
    ax2.plot(grid, perp, color="tab:red", ls="--", label="|Δv⊥| (m/s)")
    ax2.plot(grid, approche, color="tab:green", ls=":", label="vitesse de rapprochement après manoeuvre (m/s)")
    ax.set(xlabel="cap après manoeuvre φ (degrés)", ylabel="√BCR position finale (m)")
    ax2.set_ylabel("m/s")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center"); ax.grid(alpha=.3)
    fig.savefig(FIG_DIR + "/q1_geometrie.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Q2
# Q2b-(1) = experience A : Q nul pour le monde, l'EKF et la borne
# Q2b-(2) = experience B : Q non nul pour le monde et l'EKF, borne calculee avec Q = 0 (mal specifiee)
# Q2b-(3) = experience C : Q non nul pour le monde, l'EKF et la borne (coherent)
Q2_RUNS, Q2_MC, Q2_STEPS = 500, 15000, 30


def q2_simuler(runs=Q2_RUNS, mc=Q2_MC, steps=Q2_STEPS):
    """
    Simule une seule fois les trois experiences (memes graines que experiments.main)
    B et C partagent les memes trajectoires, mesures et resultats EKF : seule la borne change
    """
    zero = np.zeros_like(ex.Q)
    tA, yA = ex.simulate(runs, steps, zero, seed=2026)
    eA, cA = ex.run_ekf(tA, yA, zero)
    bA = ex.information_curve(steps, zero, mc, seed=2027)
    tBC, yBC = ex.simulate(runs, steps, ex.Q, seed=3030)
    eBC, cBC = ex.run_ekf(tBC, yBC, ex.Q)
    bB = ex.information_curve(steps, zero, mc, seed=2027)
    bC = ex.information_curve(steps, ex.Q, mc, seed=3031)
    return {"A": (tA, eA, cA, bA), "B": (tBC, eBC, cBC, bB), "C": (tBC, eBC, cBC, bC)}


def q2_figures(data):
    """
    Figures du TP (equivalent de `python experiments.py --output-dir figures`), sans resimuler
    """
    titres = {"A": "Q2b-(1) A", "B": "Q2b-(2) B", "C": "Q2b-(3) C"}     # titres courts (le detail est en legende du rapport)
    for name, (t, e, c, b) in data.items():
        ex.plot_experiment(titres[name], t, e, c, b, os.path.join(FIG_DIR, f"experience_{name}.png"))
    tBC, eBC, _, bB = data["B"]
    bC = data["C"][3]
    k = np.arange(tBC.shape[1]) * ex.DT
    rmse = np.sqrt(np.mean((eBC - tBC) ** 2, axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for i, lab in enumerate(("position (m)", "vitesse (m/s)")):
        axes[i].plot(k, 3 * np.sqrt(bB[:, i, i]), color="tab:orange", label="B : 3√BCR (Q = 0)")
        axes[i].plot(k, 3 * np.sqrt(bC[:, i, i]), color="tab:green", label="C : 3√BCR (Q > 0)")
        axes[i].plot(k, 3 * rmse[:, i], color="tab:red", label="EKF commun B/C : 3×RMSE")
        axes[i].set(xlabel="temps (s)", ylabel=lab, title="Q2b-(2) et (3) : même EKF, deux bornes")
        axes[i].grid(alpha=.25); axes[i].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "comparison_B_C.png"), dpi=160); plt.close(fig)


def q2_tableau(data, instants=(5, 15, 30)):
    """
    Valeurs 3 sigma de la borne, de l'erreur reelle et de l'incertitude annoncee par l'EKF
    """
    print(f"{'exp.':<5} {'k':>3} | {'3sqrtBCR s':>10} {'3RMSE s':>8} {'3sqrtP s':>9} | {'3sqrtBCR v':>10} {'3RMSE v':>8} {'3sqrtP v':>9} | dans 3sigma EKF")
    for name, (t, e, c, b) in data.items():
        err = e - t
        for k in instants:
            rmse = np.sqrt(np.mean(err[:, k] ** 2, axis=0))
            p = np.sqrt(np.mean(np.diagonal(c[:, k], axis1=1, axis2=2), axis=0))
            inside = 100 * np.mean(np.abs(err[:, k, 0]) <= 3 * np.sqrt(c[:, k, 0, 0]))
            print(f"{name:<5} {k:>3} | {3*np.sqrt(b[k,0,0]):10.2f} {3*rmse[0]:8.2f} {3*p[0]:9.2f} | {3*np.sqrt(b[k,1,1]):10.3f} {3*rmse[1]:8.3f} {3*p[1]:9.3f} | {inside:5.1f} %")
    tA, eA, cA, _ = data["A"]
    final = np.abs(eA[:, -1, 0] - tA[:, -1, 0])
    print(f"A, k=30 : erreur |s| mediane {np.median(final):.2f} m, moyenne quadratique {np.sqrt(np.mean(final**2)):.2f} m, runs > 3 sigma EKF : {100*np.mean(final > 3*np.sqrt(cA[:, -1, 0, 0])):.1f} %")


def q2_rapports(data):
    """
    Efficacite (RMSE / sqrt(BCR), ideal 1) et consistance (sqrt(P EKF) / RMSE, ideal 1) au cours du temps
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True)
    couleurs = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for name, (t, e, c, b) in data.items():
        k = np.arange(t.shape[1]) * ex.DT
        rmse = np.sqrt(np.mean((e - t) ** 2, axis=0))
        p = np.sqrt(np.mean(np.diagonal(c, axis1=2, axis2=3), axis=0))
        for i in range(2):
            axes[0, i].semilogy(k[1:], rmse[1:, i] / np.sqrt(b[1:, i, i]), color=couleurs[name], label=name)
            if name != "B":                                    # B et C : meme EKF, la consistance est identique
                axes[1, i].plot(k[1:], p[1:, i] / rmse[1:, i], color=couleurs[name], label=name if name == "A" else "B / C")
    for i, lab in enumerate(("position", "vitesse")):
        axes[0, i].axhline(1, color="k", lw=.8)
        axes[0, i].set(ylabel="RMSE / √BCR", title=f"Efficacité de l'EKF ({lab}) : 1 = borne atteinte")
        axes[1, i].axhline(1, color="k", lw=.8)
        axes[1, i].set(xlabel="temps (s)", ylabel="√P_EKF / RMSE", title=f"Consistance de l'EKF ({lab}) : < 1 = trop confiant")
        for j in range(2):
            axes[j, i].grid(alpha=.3, which="both"); axes[j, i].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "q2_rapports.png"), dpi=150); plt.close(fig)


def q2_consistance(data):
    """
    Erreurs normalisees par l'incertitude annoncee : histogramme a la fin, et NEES moyen au cours du temps
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    z_grid = np.linspace(-6, 6, 300)
    for name, color in (("A", "tab:blue"), ("C", "tab:green")):
        t, e, c, _ = data[name]
        err = e - t
        z = err[:, -1, 0] / np.sqrt(c[:, -1, 0, 0])
        axes[0].hist(np.clip(z, -6, 6), bins=60, range=(-6, 6), density=True, alpha=.45, color=color,
                     label=f"{name} : écart-type de z = {z.std():.2f}")
        nees = np.einsum("rki,rkij,rkj->rk", err, np.linalg.inv(c), err)          # (e^T P^-1 e) par run et par instant
        k = np.arange(t.shape[1]) * ex.DT
        axes[1].semilogy(k[1:], nees[:, 1:].mean(axis=0), color=color, label=f"{name}")
        print(f"{name} : k=30, ecart-type de l'erreur normalisee = {z.std():.2f} (ideal 1), "
              f"part hors +-3 = {100*np.mean(np.abs(z) > 3):.1f} %, NEES moyen = {nees[:, -1].mean():.2f} (ideal 2)")
    axes[0].plot(z_grid, np.exp(-z_grid**2 / 2) / np.sqrt(2*np.pi), "k", lw=1.2, label="N(0,1) (filtre consistant)")
    axes[0].set(xlabel="erreur / √P_EKF en position, k = 30", ylabel="densité", title="Erreur normalisée en fin de simulation")
    axes[0].legend(fontsize=8)
    n = data["A"][0].shape[0]
    lo, hi = stats_chi2_band(n)
    axes[1].axhspan(lo, hi, color="gray", alpha=.2, label="zone attendue (95 %)")
    axes[1].axhline(2, color="k", lw=.8)
    axes[1].set(xlabel="temps (s)", ylabel="NEES moyen", title="NEES moyen (attendu : 2 = dimension de l'état)")
    axes[1].grid(alpha=.3, which="both"); axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "q2_consistance.png"), dpi=150); plt.close(fig)


def stats_chi2_band(n_runs, dim=2, level=0.95):
    """
    Intervalle de confiance du NEES moyen sur n_runs simulations (n_runs * NEES ~ chi2(n_runs * dim))
    """
    from scipy.stats import chi2
    return (chi2.ppf((1 - level) / 2, n_runs * dim) / n_runs, chi2.ppf((1 + level) / 2, n_runs * dim) / n_runs)


def q2_apriori_serre(runs=Q2_RUNS, mc=Q2_MC, steps=Q2_STEPS):
    """
    Experience A, mais avec une incertitude initiale 10 fois plus petite en ecart-type
    """
    P0_saved = ex.P0.copy()
    out = {}
    for name, P0 in [("P0 du TP : σ_s = 10 m, σ_v = 1 m/s", P0_saved), ("P0 / 100 : σ_s = 1 m, σ_v = 0.1 m/s", P0_saved / 100)]:
        ex.P0 = P0
        zero = np.zeros_like(ex.Q)
        t, y = ex.simulate(runs, steps, zero, seed=2026)
        e, c = ex.run_ekf(t, y, zero)
        b = ex.information_curve(steps, zero, mc, seed=2027)
        rmse = np.sqrt(np.mean((e - t) ** 2, axis=0))
        out[name] = (rmse, b, np.sqrt(np.mean(np.diagonal(c, axis1=2, axis2=3), axis=0)))
        print(f"{name} : k=30, RMSE s / sqrt(BCR s) = {rmse[-1,0]/np.sqrt(b[-1,0,0]):.2f}, RMSE v / sqrt(BCR v) = {rmse[-1,1]/np.sqrt(b[-1,1,1]):.2f}")
    ex.P0 = P0_saved
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    t = np.arange(steps + 1)
    for (name, (rmse, b, p)), ls in zip(out.items(), ["-", "--"]):
        for i, ax in enumerate(axes):
            ax.plot(t, rmse[:, i] / np.sqrt(b[:, i, i]), ls=ls, color="tab:red", label=f"RMSE / √BCR, {name}")
            ax.plot(t, p[:, i] / np.sqrt(b[:, i, i]), ls=ls, color="tab:blue", label=f"√P EKF / √BCR, {name}")
    for i, lab in enumerate(["position", "vitesse"]):
        axes[i].axhline(1, color="k", lw=0.8)
        axes[i].set(xlabel="temps (s)", ylabel="rapport à la borne", title=f"Expérience A, {lab}")
        axes[i].grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, "q2_apriori_serre.png"), dpi=150); plt.close(fig)


def q2_reglage_Q(facteurs=(0.01, 0.1, 1, 10, 100), runs=Q2_RUNS, mc=Q2_MC, steps=Q2_STEPS):
    """
    Monde et borne avec le vrai Q (experience C), l'EKF utilise Q_filtre = facteur x Q
    """
    t, y = ex.simulate(runs, steps, ex.Q, seed=3030)
    b = ex.information_curve(steps, ex.Q, mc, seed=3031)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    k = np.arange(steps + 1) * ex.DT
    for f, color in zip(facteurs, plt.cm.viridis(np.linspace(0, .9, len(facteurs)))):
        e, c = ex.run_ekf(t, y, f * ex.Q)
        err = e - t
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        inside = 100 * np.mean(np.abs(err[:, :, 0]) <= 3 * np.sqrt(c[:, :, 0, 0]), axis=0)
        for i in range(2):
            axes[i].semilogy(k[1:], rmse[1:, i] / np.sqrt(b[1:, i, i]), color=color, label=f"Q_EKF = {f:g} Q")
        axes[2].plot(k[1:], inside[1:], color=color, label=f"Q_EKF = {f:g} Q")
        print(f"Q_EKF = {f:6g} Q : k=30, RMSE/sqrt(BCR) position {rmse[-1,0]/np.sqrt(b[-1,0,0]):5.2f}, vitesse {rmse[-1,1]/np.sqrt(b[-1,1,1]):5.2f}, dans +-3 sigma {inside[-1]:5.1f} %")
    for i, lab in enumerate(("position", "vitesse")):
        axes[i].axhline(1, color="k", lw=.8)
        axes[i].set(xlabel="temps (s)", ylabel="RMSE / √BCR (vrai Q)", title=f"Réglage de Q dans l'EKF : {lab}")
        axes[i].grid(alpha=.3, which="both"); axes[i].legend(fontsize=8)
    axes[2].set(xlabel="temps (s)", ylabel="% des simulations dans ±3σ (position)", title="Consistance", ylim=(0, 102))
    axes[2].grid(alpha=.3); axes[2].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "q2_reglage_Q.png"), dpi=150); plt.close(fig)


def q2_convergence_mc(tailles=(100, 300, 1000, 3000, 15000), repetitions=10, steps=Q2_STEPS):
    """
    Precision du Monte-Carlo de la borne (experience C) : borne finale selon le nombre d'echantillons
    """
    res = np.array([[3 * np.sqrt(ex.information_curve(steps, ex.Q, n, seed=100 + r)[-1, 0, 0]) for r in range(repetitions)] for n in tailles])
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.errorbar(tailles, res.mean(axis=1), yerr=res.std(axis=1), fmt="o-", capsize=4)
    ax.set_xscale("log")
    ax.set(xlabel="nombre d'échantillons Monte-Carlo", ylabel="3√BCR position à k = 30 (m)", title=f"Convergence du Monte-Carlo de la borne ({repetitions} tirages)")
    ax.grid(alpha=.3, which="both")
    fig.savefig(os.path.join(FIG_DIR, "q2_convergence_mc.png"), dpi=150); plt.close(fig)
    for n, r in zip(tailles, res):
        print(f"mc = {n:6d} : 3 sqrt(BCR) = {r.mean():.3f} m +- {r.std():.3f} m ({100*r.std()/r.mean():.2f} %)")


# ----------------------------------------------------------------------------------------------- lancement
def run_q1():
    print("=== Q1 : cap optimal (pas de 0.1 deg)"); q1_grille(0.1)
    print("=== Q1 : phi = 0 et observabilite"); q1_non_observable()
    print("=== Q1 : instant de manoeuvre"); q1_instant_manoeuvre()
    print("=== Q1 : geometrie"); q1_geometrie()


def run_q2():
    print("=== Q2 : simulation des experiences A / B / C"); data = q2_simuler()
    print("=== Q2 : figures du TP"); q2_figures(data)
    print("=== Q2 : valeurs A / B / C"); q2_tableau(data)
    print("=== Q2 : efficacite et consistance"); q2_rapports(data); q2_consistance(data)
    print("=== Q2 : experience A avec un a priori serre"); q2_apriori_serre()
    print("=== Q2 : reglage de Q dans l'EKF"); q2_reglage_Q()
    print("=== Q2 : convergence du Monte-Carlo de la borne"); q2_convergence_mc()


if __name__ == "__main__":
    import sys
    choix = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"      # python tp3_analyses.py [Q1 | Q2 | ALL]
    if choix in ("Q1", "ALL"):
        run_q1()
    if choix in ("Q2", "ALL"):
        run_q2()