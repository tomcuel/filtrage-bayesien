import numpy as np


class KalmanFilter:
    def __init__(self, x0, P0): 
        """
        x0: initial state estimate (numpy array)
        P0: initial covariance estimate (numpy array)
        """
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, F, Q):
        """
        F: state transition matrix (numpy array)
        Q: process noise covariance matrix (numpy array)
        
        Prediction:
            x_k|k-1 = F x_k-1
            P_k|k-1 = F P_k-1 F.T + Q
        """
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, y, H, R):
        """
        y: measurement (numpy array)
        H: measurement matrix (numpy array)
        R: measurement noise covariance matrix (numpy array)

        Correction:
            innovation = y - H x
            S = H P H.T + R
            K = P H.T S^-1
        """
        innovation = y - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ innovation
        self.P -= K @ H @ self.P

        return innovation, S, K
