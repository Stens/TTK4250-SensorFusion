import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import estimationstatistics as estats


def plot_cov_ellipse2d(
    ax: plt.Axes,
    mean: np.ndarray = np.zeros(2),
    cov: np.ndarray = np.eye(2),
    n_sigma: float = 1,
    *,
    edgecolor: "Color" = "C0",
    facecolor: "Color" = "none",
    **kwargs,  # extra Ellipse keyword arguments
) -> matplotlib.patches.Ellipse:
    """Plot a n_sigma covariance ellipse centered in mean into ax."""
    ell_trans_mat = np.zeros((3, 3))
    ell_trans_mat[:2, :2] = np.linalg.cholesky(cov)
    ell_trans_mat[:2, 2] = mean
    ell_trans_mat[2, 2] = 1

    ell = matplotlib.patches.Ellipse(
        (0.0, 0.0),
        2.0 * n_sigma,
        2.0 * n_sigma,
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs,
    )
    trans = matplotlib.transforms.Affine2D(ell_trans_mat)
    ell.set_transform(trans + ax.transData)
    return ax.add_patch(ell)


def run_plots(Z, Xgt, Ts, ownship, K, tracker, init_imm_state, imm_filter, models=["CV", "CT", "CV_H"]):

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
    NEES = np.zeros(K+1)
    NEESpos = np.zeros(K+1)
    NEESvel = np.zeros(K+1)

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
    poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1)
    velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1)
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

    #  plots
    # trajectory
    fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
    axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
    axs3[0].plot(*Xgt.T[:2], label="$x$")
    axs3[0].set_title(
        f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
    )
    axs3[0].axis("equal")
    # probabilities
    for j, mod in enumerate(models):
        axs3[1].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)],
                     prob_hat[:, j], label=mod)
    axs3[1].set_ylim([0, 1])
    axs3[1].set_ylabel("mode probability")
    axs3[1].set_xlabel("time")
    axs3[1].legend()

    # NEES
    fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
    axs4[0].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)], NEESpos)
    axs4[0].plot([0, np.sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    axs4[0].set_ylabel("NEES pos")
    inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
    axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

    axs4[1].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)], NEESvel)
    axs4[1].plot([0, np.sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    axs4[1].set_ylabel("NEES vel")
    inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
    axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

    axs4[2].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)], NEES)
    axs4[2].plot([0, np.sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs4[2].set_ylabel("NEES")
    inCI = np.mean((CI4[0] <= NEES) * (NEES <= CI4[1]))
    axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

    print(
        f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(
        f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

    # errors
    fig5, axs5 = plt.subplots(2, num=5, clear=True)
    axs5[0].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)],
                 np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
    axs5[0].set_ylabel("position error")

    axs5[1].plot([np.sum(Ts[:i]) for i, _ in enumerate(Ts)],
                 np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
    axs5[1].set_ylabel("velocity error")

    plot_pause = 0.2  # lenght to pause between time steps;
    start_k = 1
    end_k = 199
    plot_range = slice(start_k, end_k)  # the range to go through
    has_plotted = False
    fig6, axs6 = plt.subplots(1, 2, num=6, clear=True)
    mode_lines = [axs6[0].plot(np.nan, np.nan, color=f"C{s}")[
        0] for s in range(2)]
    meas_sc = axs6[0].scatter(np.nan, np.nan, color="r", marker="x")
    meas_sc_true = axs6[0].scatter(np.nan, np.nan, color="g", marker="x")
    min_ax = np.vstack(Z).min(axis=0)  # min(cell2mat(Z'));
    max_ax = np.vstack(Z).max(axis=0)  # max(cell2mat(Z'));
    axs6[0].axis([min_ax[0], max_ax[0], min_ax[1], max_ax[0]])

    for k, (Zk, pred_k, upd_k, Xk) in enumerate(
        zip(
            Z[plot_range],
            tracker_predict_list[plot_range],
            tracker_update_list[plot_range],
            Xgt[:, :2]
        ),
        start_k,
    ):
        (ax.cla() for ax in axs6)
        pl = []
        cl = []
        gated = tracker.gate(Zk, pred_k)
        cond_upd_k = tracker.conditional_update(Zk[gated], pred_k)
        beta_k = tracker.association_probabilities(Zk[gated], pred_k)
        for s in range(2):
            mode_lines[s].set_data = (
                np.array([u.components[s].mean[:2]
                          for u in tracker_update_list[:k]]).T,
            )
            axs6[1].plot(prob_hat[: (k - 1), s], color=f"C{s}")
            for j, cuj in enumerate(cond_upd_k):
                alpha = 0.7 * beta_k[j] * cuj.weights[s] + 0.3
                upd_km1_s = tracker_update_list[k - 1].components[s]
                cl.append(
                    axs6[0].plot(
                        [upd_km1_s.mean[0], cuj.components[s].mean[0]],
                        [upd_km1_s.mean[1], cuj.components[s].mean[1]],
                        "--",
                        color=f"C{s}",
                        alpha=alpha,
                    )
                )

                pl.append(
                    axs6[1].plot(
                        [k - 1, k],
                        [prob_hat[k - 1, s], cuj.weights[s]],
                        color=f"C{s}",
                        alpha=alpha,
                    )
                )

            Sk = imm_filter.filters[s].innovation_cov(
                [0, 0], pred_k.components[s])

            meas_sc.set_offsets(Zk)
            if not has_plotted:  # Hacky way to get labels
                pl.append(axs6[0].scatter(
                    *Zk.T, color="r", marker="x", label="meas"))
                pl.append(axs6[0].scatter(*Xk.T, color="b",
                                          marker="v", alpha=0.3, label="true"))
                pl.append(axs6[0].scatter(*upd_km1_s.mean[:2].T,
                                          color="g", marker=".", alpha=0.3, label="pred"))
                axs6[0].legend()
                has_plotted = True
            else:
                pl.append(axs6[0].scatter(
                    *Zk.T, color="r", marker="x"))
                pl.append(axs6[0].scatter(*Xk.T, color="b",
                                          marker="v", alpha=0.3))
                pl.append(axs6[0].scatter(*upd_km1_s.mean[:2].T,
                                          color="g", marker=".", alpha=0.3))

        plt.pause(plot_pause)

    plt.show()
