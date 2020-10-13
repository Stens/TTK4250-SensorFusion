# %% imports
import scipy
import scipy.io
import scipy.stats
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda

from joyride_runtime import run_plots


# %% plot config check and style setup

# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item() - 1
Ts = loaded_data["Ts"].squeeze()
Ts = np.array([0, *Ts])
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
ownship = loaded_data["ownship"]


# plot measurements close to the trajectory

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(
        Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
# plt.show()

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.05
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)

# %% setup and track

sigma_z = 10
clutter_intensity = 3e-5
PD = 0.9
gate_size = 3

# dynamic models
sigma_a_CV = 1.3
sigma_a_CV_high = 10
sigma_a_CT = 1.6
sigma_omega = 0.02*np.pi

# markov chain


p10 = 0.9  # initvalue for mode probabilities
PI = np.array(
    [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
PI = np.array([[0.85, 0.15], [0.15, 0.85]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

mean_init = np.array([*Xgt[0, :], 0])
cov_init = np.diag([20, 20, 1, 1, 0.1]) ** 2
mode_probabilities_init = np.array([0.9, 0.1])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(
    mode_probabilities_init, [mode_states_init] * 2)
init_ekf_state = GaussParams(mean_init, cov_init)

measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
# dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=5))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
# ekf_filters.append(ekf.EKF(dynamic_models[2], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

run_plots(Z, Xgt, Ts, ownship, K, tracker, init_imm_state, imm_filter)
