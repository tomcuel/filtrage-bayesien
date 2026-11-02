"""Borne de Cramer-Rao bayesienne recursive pour dynamique lineaire gaussienne.

Modele :
    X[k+1] = F[k] X[k] + w[k], w[k] ~ N(0, Q[k])
    Y[k]   = h[k](X[k]) + v[k], v[k] ~ N(0, R[k])

Les bruits sont independants, et R[k] ne depend pas de l'etat.
L'observation h[k] peut etre non lineaire : seule sa jacobienne est requise.

J[k+1] = (F[k] J[k]^-1 F[k].T + Q[k])^-1
         + E[H[k+1](X[k+1]).T R[k+1]^-1 H[k+1](X[k+1])]

L'esperance est approchee par Monte Carlo a partir d'echantillons de la
loi *inconditionnelle* de X[k+1], fournis par le programme appelant.
"""

import numpy as np


def _sym(A):
    return 0.5 * (A + A.T)


def _as_covariance(R, dimension):
    """Accepte un scalaire pour une mesure scalaire, sinon une matrice m x m."""
    R = np.asarray(R, dtype=float)
    if R.ndim == 0:
        R = R.reshape(1, 1)
    if R.shape != (dimension, dimension):
        raise ValueError(f"R doit etre de taille ({dimension}, {dimension}), recu {R.shape}")
    return R


def expected_measurement_information(H_jacobian, R, states):
    """Estime E[H(X).T R^-1 H(X)] pour un modele de mesure quelconque.

    Parameters
    ----------
    H_jacobian : callable(x) -> array de taille (m,n)
        Jacobienne de l'observation en un etat x de taille (n,).
    R : scalaire ou array (m,m)
        Covariance du bruit de mesure, definie positive.
    states : array (N,n)
        Etats issus de la loi marginale de X[k] (PAS du posterior du filtre).

    Returns
    -------
    array (n,n) : information moyenne apportee par la mesure.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("states doit etre un tableau non vide de taille (N,n)")
    n = states.shape[1]
    H0 = np.asarray(H_jacobian(states[0]), dtype=float)
    if H0.ndim == 1:  # convenance pour une observation scalaire
        H0 = H0.reshape(1, -1)
    if H0.ndim != 2 or H0.shape[1] != n:
        raise ValueError("H_jacobian doit retourner une jacobienne de taille (m,n)")
    m = H0.shape[0]
    R = _as_covariance(R, m)
    information = np.zeros((n, n))
    for x in states:
        # COMPLETER START
        H = np.asarray(H_jacobian(x), dtype=float).reshape(m, n)
        information += H.T @ np.linalg.solve(R, H)      # H(x)^T R^-1 H(x), moyenne sur les etats
        # COMPLETER END
    return _sym(information / len(states))


class GaussianDynamicsPCRB:
    """Suit l'information bayesienne J et la borne P = J^-1.

    P0 : covariance de l'a priori initial (supposee definie positive).
    Appliquer predict(F,Q), puis update(H_jacobian,R,states) a chaque date.
    """

    def __init__(self, P0):
        P0 = np.asarray(P0, dtype=float)
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError("P0 doit etre une matrice carree")
        self.J = np.linalg.inv(P0)
        self.J = _sym(self.J)

    @property
    def bound(self):
        """Borne inferieure sur la matrice d'erreur quadratique moyenne."""
        return _sym(np.linalg.inv(self.J))

    def predict(self, F, Q):
        """Propagation par une transition lineaire et son bruit gaussien.

        Q=0 est accepte (dynamique deterministe). Une matrice F singuliere
        est aussi admise tant que la covariance predite reste inversible.
        """
        F = np.asarray(F, dtype=float)
        Q = np.asarray(Q, dtype=float)
        n = self.J.shape[0]
        # COMPLETER START
        P_pred = _sym(F @ np.linalg.inv(self.J) @ F.T + Q)   # P = F J^-1 F^T + Q
        # COMPLETER END
        self.J = _sym(np.linalg.inv(P_pred))
        return P_pred

    def update_information(self, J_measurement):
        """Ajoute une information de mesure moyenne deja calculee."""
        J_measurement = np.asarray(J_measurement, dtype=float)
        if J_measurement.shape != self.J.shape:
            raise ValueError("Dimension de l'information de mesure incorrecte")
        # COMPLETER START
        self.J = _sym(self.J + J_measurement)                # J <- J_pred + E[H^T R^-1 H]
        # COMPLETER END
        return self.bound

    def update(self, H_jacobian, R, states):
        """Correction PCRB a partir de la jacobienne de h et de R."""
        J_measurement = expected_measurement_information(H_jacobian, R, states)
        return self.update_information(J_measurement)

    def step(self, F, Q, H_jacobian, R, states_next):
        """Effectue prediction k->k+1 puis mise a jour par mesure k+1."""
        self.predict(F, Q)
        return self.update(H_jacobian, R, states_next)


# Exemple d'observation, volontairement separe du moteur PCRB.
def bearing_jacobian_1d(x, sensor_xy=(0.0, -20.0)):
    """Angle atan2(-ys, s-xs), etat x=(s,v), cible sur l'axe y=0."""
    xs, ys = sensor_xy
    s = x[0]
    # COMPLETER START
    dx, dy = s - xs, 0.0 - ys
    Jdx = -dy / (dx**2 + dy**2)       # d atan2(dy, dx) / ds = -dy / r^2
    Jdvx = 0.0                        # l'angle ne depend pas de la vitesse
    # COMPLETER END
    return np.array([[Jdx, Jdvx]])
