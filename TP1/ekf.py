import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, f, F_jacobian, Q):
        """
        f: function that computes the predicted state (callable)
        F_jacobian: function that computes the Jacobian of f (callable)
        Q: process noise covariance matrix (numpy array)

        Modèle:
            x_k = f_k(x_k-1) + w_k

        F_k = df_k/dx évaluée en x_k-1
        """
        F = F_jacobian(self.x)
        self.x = f(self.x)
        self.P = F @ self.P @ F.T + Q

    def update(self, y, h, H_jacobian, R):
        """
        y: measurement (numpy array)
        h: function that computes the predicted measurement (callable)
        H_jacobian: function that computes the Jacobian of h (callable)
        R: measurement noise covariance matrix (numpy array)

        Modèle de mesure:
            y_k = h_k(x_k) + v_k

        H_k = dh_k/dx évaluée en x_k|k-1
        """
        H = H_jacobian(self.x)
        innovation = y - h(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ innovation
        self.P -= K @ H @ self.P
        
        return innovation, S, K
