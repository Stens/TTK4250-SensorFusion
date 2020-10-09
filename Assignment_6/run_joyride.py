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
import estimationstatistics as estats

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]


# plot measurements close to the trajectory

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

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
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)

# %% setup and track

# sensor
sigma_z = 2.25
clutter_intensity = 1e-3
PD = 0.95
gate_size = 4

# dynamic models
# NB: THESE ARE CORRECT
sigma_a_CV = 0.3
sigma_a_CT = 0.1
sigma_omega = 5e-4 * np.pi

# markov chain
PI11 = 0.9
PI22 = 0.9

p10 = 0.9  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

mean_init = np.array([*Xgt[0, :], 0])
cov_init = np.diag([20, 20, 5, 5, 0.1]) ** 2  
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(
    mode_probabilities_init, [mode_states_init] * 2)

measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []
# estimate
for k, (Zk, x_true_k, Tsk) in enumerate(zip(Z, Xgt, Ts)):
    tracker_predict = tracker.predict(tracker_update, Tsk)
    tracker_update = tracker.update(Zk, tracker_predict)
    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)

    NEES[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            4)
    )

    NEESpos[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2)
    )
    NEESvel[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2, 4)
    )

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)

x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:-1, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:-1, 2:4], axis=0)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()

# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)
# %%
