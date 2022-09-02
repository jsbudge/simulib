# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:24:39 2018

@author: Jeff Budge

First try at an Unscented Kalman Filter

This is an object which takes in a mean and covariance matrix
and uses it to estimate both the state matrix of a nonlinear function
and the actual Kalman update steps.

Taken from
The Unscented Kalman Filter for Nonlinear Estimation
Eric A. Wan and Rudolph van der Merwe
Oregon Graduate Institute of Science and Technology
2000

and

Kalman and Bayesian Filters in Python
Roger R. Labbe, Jr.
2018

and

The Scaled Unscented Transformation
Simon J. Julier
IDAK Industries

INPUTS
x_o                 The initial state
p_func              Process function - input must be same length as x_o
m_func              Measurement function
dt                  Measurement interval - this means nothing at the moment
Q                   Process noise covariance
R                   Measurement noise covariance
adaptive_constant   If this is set, it's the threshold at which the adaptive noise correction routine kicks in
batch_size          Number of update steps after which the smoother is run

Basic Idea

1. Get statistics for measurement vector, i.e., for x_hat.
This should give us x_mean and Pxx, our covariance estimate of x_hat.

2. Set up sigma points using Pxx and x_mean to get a vector of 2n + 1 sigma points,
where n is the dimension of x_hat. Sigma points are normalized, along with weights.

3. Run each point through the function F(x_hat, w) to yield a set of transformed sigma points.
This function is the set of nonlinear equations that we want our x_hat transformed from.

4. Take the mean and covariance of the sigma points from the weighted average and outer product
of the transformed sigma points. This is our y_mean and Pyy.

5. From here, the Kalman filter is implemented as normal for an EKF...as far as I can tell.
"""

import numpy as np
from scipy.stats import chisquare


class UKF(object):

    def __init__(self, x_o, p_func, m_func, dt=1.0, Q=None, R=None, adaptive_constant=None, batch_size=20):
        L = len(x_o)
        self.L = L
        self.z_sz = len(m_func(x_o, dt))
        if Q is not None:
            self.Q = Q
        else:
            d = 10.0 ** oofm(abs(x_o))
            d[d == 0] = 10e-9
            self.Q = np.diag(d)
        if R is not None:
            self.R = R
        else:
            d = 10.0 ** oofm(abs(m_func(x_o, dt)))
            d[d == 0] = 10e-9
            self.R = np.diag(d)
        self.P = self.Q + 0.0
        self.Pz = self.R + 0.0
        self.x = x_o
        self.dt = dt
        self.alpha = 10e-1
        self.cov_alpha = adaptive_constant
        self.p_func = p_func
        self.m_func = m_func
        self.nm_pts = 2 * L + 1
        self.P = self.initP(x_o)

        # This stuff is necessary for the smoother
        self.tracks = {'z': [], 'x': [], 'P': [], 'update_count': 0.0, 'cpi': []}
        self.z_track = []
        self.x_track = []
        self.P_track = []
        self.batch_size = batch_size
        self.batch_counter = 0

    def genSigmas(self, x_hat, Pxx):
        # Initialize all the variables we'll need
        L = self.L
        alpha = self.alpha
        kappa = L - 3
        lambda_ = alpha ** 2 * (L + kappa) - L
        X = np.zeros((2 * L + 1, L))
        W_i = np.zeros((2 * L + 1,))
        X[0, :] = x_hat
        W_i[0] = lambda_ / (L + lambda_)
        sqrtP = np.linalg.cholesky((L + lambda_) * Pxx)

        # Calculate our weights and sigma points
        for i in range(1, L + 1):
            W_i[i] = 1.0 / (2 * (L + lambda_))
            W_i[i + L] = 1.0 / (2 * (L + lambda_))
            X[i, :] = x_hat + sqrtP[i - 1, :]
            X[i + L, :] = x_hat - sqrtP[i - 1, :]

        # scaling algorithm for W's and X's
        W_i = abs(W_i) / sum(abs(W_i))
        for i in range(X.shape[0]):
            X[i, :] = x_hat + alpha * (X[i, :] - x_hat)
            W_i[i] /= alpha ** 2
        W_i[0] += 1 - 1 / alpha ** 2

        return X, W_i

    def predict(self):
        # generate sigma points
        X, W = self.genSigmas(self.x, self.P)

        # Initialize our priors and transformed sigmas
        P_bar = np.zeros_like(self.P)
        x_bar = np.zeros_like(self.x)
        Y = np.zeros((self.nm_pts, self.L))

        # run sigmas through process model and get a mean and covariance
        for i in range(self.nm_pts):
            Y[i, :] = self.p_func(X[i, :], self.dt)
            x_bar += Y[i, :] * W[i]

        for i in range(self.nm_pts):
            P_bar += np.outer(Y[i, :] - x_bar, Y[i, :] - x_bar) * W[i]
        P_bar = P_bar + (W[0] + 3 - self.alpha ** 2) * np.outer(Y[0, :] - x_bar, Y[0, :] - x_bar) + self.Q

        # save all our current state variables for use in the update step
        self.x_bar = x_bar
        self.P_bar = P_bar
        self.Y = Y
        self.W = W
        return x_bar, P_bar

    def update(self, z, curr_cpi):
        # Initialize variables
        Z = np.zeros((self.nm_pts, self.z_sz))
        Pz = np.zeros((self.z_sz, self.z_sz))
        Pxz = np.zeros((self.L, self.z_sz))
        mu_z = np.zeros((self.z_sz,))
        S_hat = np.zeros((self.z_sz, self.z_sz))
        z_k = np.zeros((self.z_sz,))

        # calculate mean and covariance of measurement space sigmas
        for i in range(self.nm_pts):
            Z[i, :] = self.m_func(self.Y[i, :], self.dt)
            mu_z += Z[i, :] * self.W[i]

        for i in range(self.nm_pts):
            Pz += np.outer(Z[i, :] - mu_z, Z[i, :] - mu_z) * self.W[i]
        Pz += self.R

        # compute residual
        y = z - mu_z
        mu_k = np.reshape(z - self.m_func(self.x_bar, self.dt), (self.z_sz, 1))  # innovation

        # get Kalman gain using cross covariance of state and measurements
        for i in range(self.nm_pts):
            Pxz += np.outer(self.Y[i, :] - self.x_bar, Z[i, :] - mu_z) * self.W[i]
        K = Pxz.dot(np.linalg.inv(Pz))

        # compute new state and covariance
        self.x = self.x_bar + K.dot(y)
        self.P = self.P_bar - K.dot(Pz.dot(K.T))
        self.Pz = Pz

        if self.cov_alpha is not None:
            phi_k = mu_k.T.dot(np.linalg.pinv(Pz)).dot(mu_k)
            chi_val = chisquare(self.cov_alpha, self.z_sz)[0]
            if chi_val > phi_k:
                # use some adaptive noise covariances
                epsilon = z - self.m_func(self.x, self.dt)  # residual
                lam_param = max([.01, (phi_k - .01 * chi_val) / phi_k])
                del_param = max([.01, (phi_k - .01 * chi_val) / phi_k])

                # estimate S_hat
                sig_k, W_k = self.genSigmas(self.x, self.P)
                for i in range(self.nm_pts):
                    z_k += self.m_func(sig_k[i, :], self.dt) * W_k[i]
                for i in range(self.nm_pts):
                    S_hat += np.outer(self.m_func(sig_k[i, :], self.dt) - z_k,
                                      self.m_func(sig_k[i, :], self.dt) - z_k) * W_k[i]

                # update Q and R
                self.Q = (1 - lam_param) * self.Q + lam_param * (K.dot(mu_k).dot(mu_k.T).dot(K.T))
                self.R = (1 - del_param) * self.R + del_param * (np.outer(epsilon, epsilon) + S_hat)

                # Do the correction step afterwards, using the new Q and R
                self.correction(z, mu_z, S_hat)

        self.tracks['x'].append(self.x)
        self.tracks['z'].append(self.getMeasurement())
        self.tracks['P'].append(self.P)
        self.tracks['cpi'].append(curr_cpi)
        # self.tracks['update_count'] += 1
        self.batch_counter += 1
        if self.batch_counter % self.batch_size == 0:
            self.smoother()
            # self.batch_counter = 0

    def correction(self, z, mu_z, S):
        print('Entered correction')
        P_bar = np.zeros_like(self.P)
        Z = np.zeros((self.nm_pts, self.z_sz))
        mu_z = np.zeros((self.z_sz,))
        Pxz = np.zeros((self.L, self.z_sz))

        for i in range(self.nm_pts):
            P_bar += np.outer(self.Y[i, :] - self.x, self.Y[i, :] - self.x) * self.W[i]
        P_bar = P_bar + (self.W[0] + 3 - self.alpha ** 2) * np.outer(self.Y[0, :] - self.x,
                                                                     self.Y[0, :] - self.x) + self.Q

        # calculate mean and covariance of measurement space sigmas
        for i in range(self.nm_pts):
            Z[i, :] = self.m_func(self.Y[i, :], self.dt)
            mu_z += Z[i, :] * self.W[i]

        Pz = S + self.R
        # get Kalman gain using cross covariance of state and measurements
        for i in range(self.nm_pts):
            Pxz += np.outer(self.Y[i, :] - self.x, Z[i, :] - mu_z) * self.W[i]
        K = Pxz.dot(np.linalg.inv(Pz))

        # compute new state and covariance
        self.x = self.x + K.dot(z - mu_z)
        self.P = P_bar - K.dot(Pz.dot(K.T))
        self.Pz = Pz

    def getMeasurement(self):
        return self.m_func(self.x, self.dt)

    def getPredictedMeasurement(self):
        return self.m_func(self.x_bar, self.dt)

    def initP(self, x_o):
        P = np.zeros((self.L, self.L))
        for _ in range(1000):
            x, Pt = self.predict()
            P += Pt
            self.x = x_o
        return P / 1000

    def smoother(self):
        curr_idx = len(self.tracks['x'])
        for i in range(curr_idx - 1, 0):
            # predict
            P = self.Pz.dot(self.tracks['P'][i]).dot(self.Pz.T) + self.Q

            # update
            Kk = self.tracks['P'][i].dot(self.Pz.T).dot(np.linalg.inv(P))
            self.tracks['x'][i] = self.tracks['x'][i] + Kk.dot(
                self.tracks['x'][i + 1] - self.Pz.dot(self.tracks['x'][i]))
            self.tracks['P'][i] = self.tracks['P'][i] + Kk.dot(self.tracks['P'][i + 1] - P).dot(Kk.T)


"""

Josh model.
Constant velocity model, with the acceleration falling out in the noise matrices.

"""


def process_function(state_vec, dt):
    proc = np.zeros((len(state_vec),))
    proc[0] = state_vec[0] + state_vec[4]
    proc[1] = state_vec[1] + state_vec[5]
    proc[2] = state_vec[2] + state_vec[6]
    proc[3] = state_vec[3] + state_vec[7]
    proc[4] = state_vec[4]
    proc[5] = state_vec[5]
    proc[6] = state_vec[6]
    proc[7] = state_vec[7]
    return proc


def observation_function(state_vec, dt):
    proc = np.zeros((5,))
    proc[0] = state_vec[0]
    proc[1] = state_vec[1]
    proc[2] = state_vec[2]
    proc[3] = state_vec[3]
    proc[4] = state_vec[7]
    return proc


"""

Angle based motion model.
This tends to oscillate, so probably not the best model to use.

"""


# def process_function(state_vec, dt):
#    proc = np.zeros((len(state_vec),))
#    r1 = state_vec[2] + state_vec[5]
#    theta1 = state_vec[0] + state_vec[3]
#    phi1 = state_vec[1] + state_vec[4]
#    x_t = r1 * np.cos(theta1) * np.cos(phi1)
#    y_t = r1 * np.cos(theta1) * np.sin(phi1)
#    z_t = r1 * np.sin(theta1)
#    r_t = np.sqrt(x_t**2 + y_t**2 + z_t**2)
#    proc[0] = np.arcsin(z_t / r_t)
#    proc[1] = np.arctan2(y_t , x_t)
#    proc[2] = r_t
#    proc[3] = proc[0] - state_vec[0]
#    proc[4] = proc[1] - state_vec[1]
#    proc[5] = proc[2] - state_vec[2]
#    return proc
#
# def observation_function(state_vec, dt):
#    #this takes the observed and updated state vector and turns it into Cartesian coordinates - northing, easting, altitude
#    proc = np.zeros((5,))
#    proc[0] = state_vec[2] * np.cos(state_vec[0]) * np.cos(state_vec[1])
#    proc[1] = state_vec[2] * np.cos(state_vec[0]) * np.sin(state_vec[1])
#    proc[2] = state_vec[2] * np.sin(state_vec[0])
#    proc[3] = state_vec[2]
#    proc[4] = state_vec[5]
#    return proc

def oofm(x):
    return np.floor(np.log10(x)).astype(np.int)
