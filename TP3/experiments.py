"""Three TP experiences comparing bearing-only EKF and posterior CRB.

A: Q_true=Q_filter=Q_bound=0
B: Q_true=Q_filter=Q_nonzero, but Q_bound=0 (misspecified bound)
C: Q_true=Q_filter=Q_bound=Q_nonzero

B/C reuse exactly the same simulated trajectories, measurements and EKF.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from ekf import ExtendedKalmanFilter
from cramer_rao_general import GaussianDynamicsPCRB


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


DT = 1.0
F = np.array([[1., DT], [0., 1.]])
Q = np.diag([0.0001, 0.01])
R = np.array([[np.deg2rad(1.)**2]])
SENSOR = np.array([0., -20.])
X0_MEAN = np.array([0., 5.])
P0 = np.diag([100., 1.])


def f(x):
    return F @ x


def f_jac(x):
    return F


def h(x):
    return np.array([np.arctan2(-SENSOR[1], x[0] - SENSOR[0])])


def h_jac(x):
    s = x[0]
    xs, ys = SENSOR
    dx, dy = s - xs, 0.0 - ys
    Jdx = -dy / (dx**2 + dy**2)       # d/ds atan2(-ys, s - xs) = ys / ((s-xs)^2 + ys^2)
    Jdvx = 0.0
    return np.array([[Jdx, Jdvx]])


def simulate(n_runs, n_steps, process_Q, seed):
    rng = np.random.default_rng(seed)
    truth = np.empty((n_runs, n_steps+1, 2))
    observed = np.empty((n_runs, n_steps+1))
    truth[:, 0, :] = rng.multivariate_normal(X0_MEAN, P0, size=n_runs)
    observed[:, 0] = h_batch(truth[:, 0, :]) + rng.normal(0, np.sqrt(R[0,0]), size=n_runs)
    noise_std = np.sqrt(np.diag(process_Q))
    for k in range(1, n_steps+1):
        truth[:, k, :] = truth[:, k-1, :] @ F.T + rng.normal(size=(n_runs,2))*noise_std
        observed[:, k] = h_batch(truth[:, k, :]) + rng.normal(0, np.sqrt(R[0,0]), size=n_runs)
    return truth, observed


def h_batch(x):
    return np.arctan2(-SENSOR[1], x[...,0] - SENSOR[0])


def run_ekf(truth, observed, Q_filter):
    n_runs, t_count, _ = truth.shape
    estimates = np.empty_like(truth)
    covariances = np.empty((n_runs, t_count, 2, 2))
    for j in range(n_runs):
        filter_ = ExtendedKalmanFilter(X0_MEAN, P0)
        estimates[j,0] = filter_.x
        covariances[j,0] = filter_.P
        for k in range(1,t_count):
            filter_.predict(f, f_jac, Q_filter)
            filter_.update(np.array([observed[j,k]]), h, h_jac, R,
                           innovation_function=wrap)
            estimates[j,k] = filter_.x
            covariances[j,k] = filter_.P
    return estimates, covariances


def information_curve(n_steps, bound_Q, mc_samples, seed):
    """MC sur la loi marginale de X_k sous le modele suppose pour la borne.

    Q=0 et Q>0 utilisent la meme classe de BCR.
    La geometrie de la mesure est definie par h_jac hors de la librairie.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(X0_MEAN, P0, size=mc_samples)
    bound_calc = GaussianDynamicsPCRB(P0)
    bounds = np.empty((n_steps+1,2,2))
    bounds[0] = P0
    noise_std = np.sqrt(np.diag(bound_Q))
    for k in range(1,n_steps+1):
        # echantillons de la loi marginale de X_k : on propage avec la dynamique supposee par la borne
        samples = samples @ F.T + rng.normal(size=samples.shape) * noise_std
        bounds[k] = bound_calc.step(
            F=F, Q=bound_Q, H_jacobian=h_jac, R=R, states_next=samples
        )
    return bounds


def plot_experiment(name, truth, est, cov, bcr, filename):
    errors = est - truth
    mse = np.mean(errors**2,axis=0)
    ekf_variance = np.mean(np.diagonal(cov,axis1=2,axis2=3), axis=0)
    bcr_variance = np.diagonal(bcr,axis1=1,axis2=2)
    time = np.arange(truth.shape[1])*DT
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i, label in enumerate(('position s (m)','vitesse v (m/s)')):
        ax = axes[0,i]
        ax.plot(time,3*np.sqrt(bcr_variance[:,i]),label='3√BCR',color='tab:green')
        ax.plot(time,3*np.sqrt(mse[:,i]),label='3×RMSE EKF',color='tab:red')
        ax.plot(time,3*np.sqrt(ekf_variance[:,i]),label='3√(moyenne P EKF)',color='tab:blue',ls='--')
        ax.set(xlabel='temps (s)',ylabel=label,title=f'{name} — incertitude {label}')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    ax = axes[1,0]
    ax.plot(time,bcr[:,0,1],label='BCR(s,v)',color='tab:green')
    ax.plot(time,np.mean(cov[:,:,0,1],axis=0),label='moyenne P_EKF(s,v)',color='tab:blue',ls='--')
    empirical_cross = np.mean(errors[:,:,0]*errors[:,:,1],axis=0)
    ax.plot(time,empirical_cross,label='E[erreur_s erreur_v]',color='tab:red')
    ax.set(xlabel='temps (s)',ylabel='covariance (m²/s)',title='Terme croisé (signé)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax = axes[1,1]
    corr = bcr[:,0,1] / np.sqrt(bcr[:,0,0]*bcr[:,1,1])
    ax.plot(time,corr,label='corrélation de la BCR',color='tab:purple')
    ax.set(xlabel='temps (s)',ylabel='corrélation',ylim=(-1.05,1.05),title='Terme croisé normalisé')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(filename,dpi=160)
    plt.close(fig)
    return mse, bcr_variance


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=int,default=500)
    parser.add_argument('--mc',type=int,default=15000)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--output-dir',type=str,default='.',help='dossier des figures (cree si besoin)')
    args=parser.parse_args(argv)
    os.makedirs(args.output_dir,exist_ok=True)
    out=lambda name: os.path.join(args.output_dir,name)

    zero=np.zeros_like(Q)
    trueA,yA=simulate(args.runs,args.steps,zero,seed=2026)
    estA,covA=run_ekf(trueA,yA,zero)
    bcrA=information_curve(args.steps,zero,args.mc,seed=2027)
    plot_experiment('A (modèle idéal)',trueA,estA,covA,bcrA,out('experience_A.png'))

    trueBC,yBC=simulate(args.runs,args.steps,Q,seed=3030)
    estBC,covBC=run_ekf(trueBC,yBC,Q)
    bcrB=information_curve(args.steps,zero,args.mc,seed=2027)
    bcrC=information_curve(args.steps,Q,args.mc,seed=3031)
    plot_experiment('B (borne mal spécifiée)',trueBC,estBC,covBC,bcrB,out('experience_B.png'))
    plot_experiment('C (modèle cohérent)',trueBC,estBC,covBC,bcrC,out('experience_C.png'))

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i, label in enumerate(('position (m)','vitesse (m/s)')):
        t=np.arange(args.steps+1)
        rmse=np.sqrt(np.mean((estBC-trueBC)**2,axis=0)[:,i])
        axes[i].plot(t,3*np.sqrt(bcrB[:,i,i]),label='B : 3√BCR(Q=0)',color='tab:orange')
        axes[i].plot(t,3*np.sqrt(bcrC[:,i,i]),label='C : 3√BCR(Q>0)',color='tab:green')
        axes[i].plot(t,3*rmse,label='EKF commun B/C : 3×RMSE',color='tab:red')
        axes[i].set(xlabel='temps (s)',ylabel=label,title='Comparaison B et C')
        axes[i].grid(alpha=.25)
        axes[i].legend(fontsize=8)
    fig.savefig(out('comparison_B_C.png'),dpi=160)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print('Figures créées : experience_A.png, experience_B.png, experience_C.png, comparison_B_C.png')
    print('B/C : mêmes trajectoires, mesures et résultats EKF.')
    print('Attention : BCR(B) n’est pas une borne de l’erreur pour le modèle vrai Q>0.')


if __name__ == '__main__':
    main()