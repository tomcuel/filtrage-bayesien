"""Q1 : choix du cap de l'observateur par evaluation sur une grille.

Modele issu des diapos 23-28 : cible a vitesse constante non bruitee,
mesure d'azimut (sigma=1 deg), observateur a 10 m/s initialement selon +x.
Convention explicite : changement instantane de cap au milieu des mesures,
cap apres manoeuvre = phi par rapport a l'axe +x, phi en degres.

La BCR est ici une approximation LOCALE sur la trajectoire nominale.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from cramer_rao_deterministe import DeterministicDynamicsPCRB

DT = 1.0
N_MEASURES = 100
SIGMA_ANGLE = np.deg2rad(1.0)
SPEED_OBSERVER = 10.0
SPEED_TARGET = 5.0
HEADING_TARGET = np.deg2rad(-20.0)
X0_TARGET = np.array([2000., 2000., SPEED_TARGET*np.cos(HEADING_TARGET), SPEED_TARGET*np.sin(HEADING_TARGET)])
P0 = np.diag(np.array([1000., 1000., 10., 10.])**2)
F = np.array([[1., 0., DT, 0.], [0., 1., 0., DT], [0., 0., 1., 0.], [0., 0., 0., 1.]])
R = np.array([[SIGMA_ANGLE**2]])


def bearing_jacobian(x, observer_xy):
    """Jacobian of atan2(y-yo,x-xo), size (1,4)."""
    dx, dy = x[0] - observer_xy[0], x[1] - observer_xy[1]
    r2 = dx**2 + dy**2
    Jdx = -dy / r2                    # d theta / dx
    Jdy = dx / r2                     # d theta / dy
    Jdvx = 0.0                        # l'angle ne depend pas des vitesses
    Jdvy = 0.0
    return np.array([[Jdx, Jdy, Jdvx, Jdvy]])


def observer_trajectory(phi_deg, n_measures=N_MEASURES, manoeuvre_index=None):
    """Positions d'observateur de t=0 a t=(n_measures-1)*DT.

    Le changement de cap a lieu APRES la mesure d'indice manoeuvre_index-1.
    """
    if manoeuvre_index is None:
        manoeuvre_index = n_measures // 2
    if not (0 <= manoeuvre_index <= n_measures):
        raise ValueError('manoeuvre_index hors limites')
    phi = np.deg2rad(phi_deg)
    v_after = SPEED_OBSERVER * np.array([np.cos(phi), np.sin(phi)])
    p = np.zeros((n_measures, 2))
    for k in range(1, n_measures):
        velocity = np.array([SPEED_OBSERVER, 0.]) if k < manoeuvre_index else v_after
        p[k] = p[k-1] + DT * velocity
    return p


def evaluate_phi(phi_deg, n_measures=N_MEASURES, manoeuvre_index=None, include_initial_measure=True):
    """Retourne critere position et toute la suite des bornes (n,4,4)."""
    observer_xy = observer_trajectory(phi_deg, n_measures, manoeuvre_index)
    bcr = DeterministicDynamicsPCRB(P0)
    trajectory = np.empty((n_measures, 4))
    trajectory[0] = X0_TARGET
    bounds = np.empty((n_measures, 4, 4))
    if include_initial_measure:
        bounds[0] = bcr.update(bearing_jacobian(trajectory[0], observer_xy[0]), R)
    else:
        bounds[0] = P0
    for k in range(1, n_measures):
        trajectory[k] = F @ trajectory[k-1]
        bounds[k] = bcr.step(F, bearing_jacobian(trajectory[k], observer_xy[k]), R)
    # somme des variances de position = erreur quadratique 2D
    criterion = float(np.trace(bounds[-1, :2, :2]))
    return criterion, bounds, trajectory, observer_xy


def main(argv=None):
    """argv : liste d'arguments (None = ligne de commande), pour pouvoir appeler main depuis un autre script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step-deg', type=float, default=1.0)
    parser.add_argument('--n-measures', type=int, default=N_MEASURES)
    parser.add_argument('--manoeuvre-index', type=int, default=None)
    parser.add_argument('--output-prefix', type=str, default='q1')
    parser.add_argument('--output-dir', type=str, default='.', help='dossier des figures (cree si besoin)')
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.output_prefix)
    if not (args.step_deg > 0):
        parser.error('--step-deg doit etre positif')
    grid = np.arange(0., 360. + 1e-10, args.step_deg)
    # 360 equivaut a 0 : ne pas evaluer le doublon
    grid = grid[grid < 360.]
    costs = np.array([evaluate_phi(p, args.n_measures, args.manoeuvre_index)[0] for p in grid])
    best = int(np.argmin(costs))
    best_phi = float(grid[best])
    criterion, bounds, target, observer = evaluate_phi(best_phi, args.n_measures, args.manoeuvre_index)
    print(f'Angle optimal sur la grille : {best_phi:.1f} deg')
    print(f'Critere position final trace(BCR_pos) = {criterion:.3f} m^2')
    print(f'RMS 2D minimale (borne locale) = {np.sqrt(criterion):.3f} m')

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.plot(grid, np.sqrt(costs), lw=1.8)
    ax.axvline(best_phi, color='tab:red', ls='--', label=f'optimum grille : {best_phi:g}°')
    ax.set(xlabel='Cap après manœuvre φ (degrés)', ylabel='√(BCR_xx + BCR_yy) (m)',
           title='Q1 — précision finale sur la position (borne locale)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(prefix + '_grille.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    ax.plot(target[:, 0], target[:, 1], label='Cible (mouvement rectiligne)', lw=2)
    ax.plot(observer[:, 0], observer[:, 1], label=f'Observateur (φ={best_phi:g}°)', lw=2)
    m = args.n_measures // 2 if args.manoeuvre_index is None else args.manoeuvre_index
    ax.scatter(observer[max(0, min(m-1, len(observer)-1)), 0],
               observer[max(0, min(m-1, len(observer)-1)), 1], marker='x', s=70, label='Manœuvre')
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Q1 — trajectoires nominales')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(prefix + '_trajectoires.png', dpi=160)
    plt.close(fig)

    t = DT*np.arange(args.n_measures)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for a, inds, label in zip(axs, ((0, 1), (2, 3)), ('Position (m)', 'Vitesse (m/s)')):
        for i, name in zip(inds, ('x', 'y')):
            a.plot(t, np.sqrt(np.maximum(bounds[:, i, i], 0.)), label=name)
        a.set(xlabel='Temps (s)', ylabel='Écart-type BCR locale', title=label)
        a.grid(alpha=.3)
        a.legend()
    fig.savefig(prefix + '_incertitudes.png', dpi=160)
    plt.close(fig)
    print('Figures :', prefix + '_grille.png,', prefix + '_trajectoires.png,', prefix + '_incertitudes.png')
    return best_phi, criterion

if __name__ == '__main__':
    main()