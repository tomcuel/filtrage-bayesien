"""
TP2 - Partie 1 : approximation de mu = int_0^1 cos(pi x / 2) dx = 2/pi par Monte-Carlo.

  1. Monte-Carlo force brute (q = U[0,1]) : variance théorique et empirique, convergence, TCL
  2. Échantillonnage d'importance avec q~(x) = 3/2 (1 - x^2), simulée par la méthode de rejet
  3. Densité d'importance optimale q*(x) = (pi/2) cos(pi x / 2)
  4. Échantillonnage d'importance avec q~(x) = 2 (1 - x), simulée par inversion de la fonction de répartition
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(0)

MU = 2 / np.pi
N = 50            # taille d'un échantillon
M = 20000         # nombre de répétitions de l'estimateur (pour mesurer sa variance)


def g(x):
    return np.cos(np.pi * x / 2)


def q(x):
    return np.ones_like(x)                     # loi uniforme sur [0, 1]


def q_poly(x):
    return 1.5 * (1 - x ** 2)                  # q~(x) = 3/2 (1 - x^2)


def q_lin(x):
    return 2 * (1 - x)                         # q~(x) = 2 (1 - x)


A_DL = 2 * np.sqrt(2) / np.pi                  # racine positive du DL 1 - pi^2 x^2 / 8


def q_dl_trunc(x):
    """DL tronqué à [0, a] puis normalisé : 3/(2a) (1 - x^2/a^2), nul sur ]a, 1]"""
    return np.where(x < A_DL, 1.5 / A_DL * (1 - (x / A_DL) ** 2), 0.0)


def q_opt(x):
    return np.pi / 2 * np.cos(np.pi * x / 2)   # densité optimale g q / mu


# générateurs
def sample_uniform(n):
    return rng.random(n), n                    # (échantillon, nombre de tirages uniformes consommés)


def sample_poly_rejection(n):
    """Rejet : proposition X ~ U[0,1], acceptation si U <= 1 - X^2 (car q~ <= 3/2 = C)"""
    out, used = [], 0
    while len(out) < n:
        x, u = rng.random(), rng.random()
        used += 2                              # deux tirages uniformes par proposition
        if u <= 1 - x ** 2:
            out.append(x)
    return np.array(out), used


def sample_dl_trunc(n):
    """Y ~ 3/2 (1 - y^2) par rejet, puis X = a Y suit le DL tronqué"""
    y, used = sample_poly_rejection(n)
    return A_DL * y, used


def acceptance_rate(n_prop=10**6):
    x, u = rng.random(n_prop), rng.random(n_prop)
    return np.mean(u <= 1 - x ** 2)


def sample_lin_inversion(n):
    """F(x) = 2x - x^2  =>  x = 1 - sqrt(1 - u)"""
    u = rng.random(n)
    return 1 - np.sqrt(1 - u), n


def sample_opt_inversion(n):
    """F(x) = sin(pi x / 2)  =>  x = (2/pi) arcsin(u)"""
    u = rng.random(n)
    return 2 / np.pi * np.arcsin(u), n


# estimateurs
def estimate(sampler, density, n):
    x, used = sampler(n)
    w = q(x) / density(x)
    return np.mean(w * g(x)), used


def theoretical_variance(density):
    """Variance pour UN échantillon : int g^2 q^2 / q~ - mu^2"""
    val, _ = integrate.quad(lambda x: g(x) ** 2 * q(np.array(x)) ** 2 / density(np.array(x)), 0, 1, limit=200)
    return float(val - MU ** 2)


def repeated(sampler, density, n=N, m=M):
    t0 = time.time()
    res = [estimate(sampler, density, n) for _ in range(m)]
    dt = time.time() - t0
    est = np.array([r[0] for r in res])
    used = np.mean([r[1] for r in res])
    return est, used, dt


if __name__ == "__main__":
    methods = {
        "force brute (q = U[0,1])": (sample_uniform, q),
        "importance 3/2(1-x²) par rejet": (sample_poly_rejection, q_poly),
        "importance 2(1-x) par inversion": (sample_lin_inversion, q_lin),
        "importance optimale (π/2)cos(πx/2)": (sample_opt_inversion, q_opt),
    }

    # 1. variances théoriques et empiriques
    print(f"mu = 2/pi = {MU:.6f}")
    print(f"{'méthode':<38} | V théo (1 éch.) | V(mu_N) théo N=50 | V(mu_N) emp. N=50 | biais emp. | unif./estim. | temps")
    results = {}
    for name, (sampler, dens) in methods.items():
        vth = theoretical_variance(dens)
        est, used, dt = repeated(sampler, dens)
        results[name] = (vth, est, used, dt)
        print(f"{name:<38} | {vth:15.3e} | {vth / N:17.3e} | {np.var(est, ddof=1):17.3e} | {np.mean(est) - MU:+.1e} | {used:12.1f} | {dt:5.2f} s")
    vb = results["force brute (q = U[0,1])"][0]
    for name in list(methods)[1:3]:
        vi = results[name][0]
        ve_ratio = np.var(results["force brute (q = U[0,1])"][1]) / np.var(results[name][1])
        print(f"rapport des variances force brute / {name} : théorique {vb / vi:.1f}, simulé {ve_ratio:.1f}")
    print(f"probabilité d'acceptation du rejet : théorique 2/3 = {2/3:.4f}, simulée {acceptance_rate():.4f}")
    print(f"racine positive de 1 - pi^2 x^2 / 8 : x = {np.sqrt(8) / np.pi:.4f}")
    C_dl = 24 / (24 - np.pi ** 2)
    print(f"DL normalisé sur [0,1] : C = 24/(24 - pi^2) = {C_dl:.4f}, valeur en x = 1 : {C_dl * (1 - np.pi**2 / 8):.3f}")
    est = np.array([estimate(sample_dl_trunc, q_dl_trunc, N)[0] for _ in range(M)])
    print(f"DL tronqué à [0, {A_DL:.4f}] : moyenne des estimations {est.mean():.5f}, valeur visée mu sin(sqrt 2) = {MU * np.sin(np.sqrt(2)):.5f}, biais {est.mean() - MU:+.5f} ({100 * (est.mean() - MU) / MU:+.2f} %)")

    # 2. convergence en fonction de N (force brute et importance 3/2(1-x²))
    ns = np.unique(np.logspace(0.5, 4, 25).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in [("force brute (q = U[0,1])", "tab:blue"), ("importance 3/2(1-x²) par rejet", "tab:orange")]:
        sampler, dens = methods[name]
        rmse = [np.sqrt(np.mean((np.array([estimate(sampler, dens, n)[0] for _ in range(400)]) - MU) ** 2)) for n in ns]
        ax.loglog(ns, rmse, "o", ms=3, color=c, label=f"{name.split(' (')[0].split(' par')[0]} (simulé)")
        ax.loglog(ns, np.sqrt(results[name][0] / ns), "-", color=c, lw=1, label=r"$\sqrt{\mathbb{V}/N}$ théorique")
    ax.set_xlabel("N"); ax.set_ylabel("erreur quadratique moyenne"); ax.set_title(r"Décroissance de l'erreur en $1/\sqrt{N}$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("figures/tp2_p1_convergence.png", dpi=120); plt.close(fig)

    # 3. densités
    xx = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xx, q(xx), label=r"$q(x)=1$ (uniforme)")
    ax.plot(xx, g(xx) * q(xx), lw=2.5, label=r"$g(x)q(x)=\cos(\pi x/2)$")
    ax.plot(xx, q_opt(xx), "--", label=r"$q^*(x)=\frac{\pi}{2}\cos(\pi x/2)$ (optimale)")
    ax.plot(xx, q_poly(xx), label=r"$\tilde q(x)=\frac{3}{2}(1-x^2)$")
    ax.plot(xx, q_lin(xx), label=r"$\tilde q(x)=2(1-x)$")
    ax.plot(xx, 1 - np.pi ** 2 * xx ** 2 / 8, ":", color="gray", label=r"DL : $1-\pi^2x^2/8$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("x"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.3, 2.1)
    plt.tight_layout(); plt.savefig("figures/tp2_p1_densites.png", dpi=120); plt.close(fig)

    # 4. TCL : histogrammes de sqrt(N)(mu_N - mu) comparés à la loi normale théorique
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, name in zip(ax, list(methods)[:3]):
        vth, est, _, _ = results[name]
        a.hist(est, bins=60, density=True, alpha=0.6, label="histogramme (20 000 estimations)")
        xs_ = np.linspace(est.min(), est.max(), 300)
        a.plot(xs_, stats.norm.pdf(xs_, MU, np.sqrt(vth / N)), "r", lw=2, label=r"$\mathcal{N}(\mu,\mathbb{V}/N)$ théorique")
        a.axvline(MU, color="k", ls="--", lw=1)
        a.set_title(name, fontsize=10); a.set_xlabel(r"$\hat\mu_{50}$"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figures/tp2_p1_tcl.png", dpi=120); plt.close(fig)

    # test de normalité (Kolmogorov-Smirnov) sur les estimations centrées réduites
    for name in list(methods)[:3]:
        vth, est, _, _ = results[name]
        z = (est - MU) / np.sqrt(vth / N)
        print(f"KS {name:<36}: p-value = {stats.kstest(z, 'norm').pvalue:.3f}, écart-type empirique de z = {z.std():.3f}")
