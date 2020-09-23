import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import indices
import scipy.stats

rng = np.random.default_rng()
# trajectory generation
# scenario parameters
x0 = np.array([np.pi / 2, -np.pi / 100])
Ts = 0.05
K = round(20 / Ts)

# constants
g = 9.81
l = 1
a = g / l
d = 0.5  # dampening
S = 5

# disturbance PDF
process_noise_sampler = lambda: rng.uniform(-S, S)

# dynamic function
def modulo2pi(x, idx=0):
    xmod = x
    xmod[idx] = (xmod[idx] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return xmod


def pendulum_dynamics(x, a, d=0):  # continuous dynamics
    xdot = np.array([x[1], -d * x[1] - a * np.sin(x[0])])
    return xdot


def pendulum_dynamics_discrete(xk, vk, Ts, a, d=0):
    xkp1 = modulo2pi(xk + Ts * pendulum_dynamics(xk, a, d))  # euler discretize
    xkp1[1] += Ts * vk  #  zero orde hold noise
    return xkp1



# sample a trajectory
x = np.zeros((K, 2))
x[0] = x0
for k in range(K - 1):
    v = process_noise_sampler()
    x[k + 1] = pendulum_dynamics_discrete(x[k], v, Ts, a, d)


# vizualize
fig1, axs1 = plt.subplots(2, sharex=True, num=1, clear=True)
axs1[0].plot(x[:, 0])
axs1[0].set_ylabel(r"$\theta$")
axs1[0].set_ylim((-np.pi, np.pi))

axs1[1].plot(x[:, 1])
axs1[1].set_xlabel("Time step")
axs1[1].set_ylabel(r"$\dot \theta$")

# measurement generation

# constants
Ld = 4
Ll = 0
r = 0.25

# noise pdf
measurement_noise_sampler = lambda: rng.triangular(-r, 0, r)

# measurement function
def h(x, Ld, l, Ll):  # measurement function
    lcth = l * np.cos(x[0])
    lsth = l * np.sin(x[0])
    z = np.sqrt((Ld - lcth) ** 2 + (lsth - Ll) ** 2)  # 2norm
    return z


Z = np.zeros(K)
for k, xk in enumerate(x):
    wk = measurement_noise_sampler()
    Z[k] = h(x[k], Ld, l, Ll) + wk


# vizualize
fig2, ax2 = plt.subplots(num=2, clear=True)
ax2.plot(Z)
ax2.set_xlabel("Time step")
ax2.set_ylabel("z")

# Task: Estimate using a particle filter

# number of particles to use
N =  200 # around 100-200 seems to be fine, best relative degeneracy with 100

# initialize particles, pretend you do not know where the pendulum starts
px = np.array([
    np.random.normal(-np.pi, np.pi, N),
    np.random.uniform(size=N)*np.pi/2
    ]).T

# initial weights
w = np.ones(N)/N # Equal weight, with sum 1
# allocate some space for resampling particles
pxn = np.zeros_like(px)

# PF transition PDF: SIR proposal, or something you would like to test
PF_dynamic_distribution = scipy.stats.uniform(loc=-S, scale=2 * S)
PF_measurement_distribution = scipy.stats.triang(c=0.5, loc=-r, scale=2 * r)

# initialize a figure for particle animation.
plt.ion()
fig4, ax4 = plt.subplots(num=4, clear=True)
plotpause = 0.01

sch_particles = ax4.scatter(np.nan, np.nan, marker=".", c="b", label=r"$\hat \theta^n$")
sch_true = ax4.scatter(np.nan, np.nan, c="r", marker="x", label=r"$\theta$")
ax4.set_ylim((-1.5 * l, 1.5 * l))
ax4.set_xlim((-1.5 * l, 1.5 * l))
ax4.set_xlabel("x")
ax4.set_ylabel("y")
th = ax4.set_title(f"theta mapped to x-y")
ax4.legend()

eps = np.finfo(float).eps
for k in range(K):
    print(f"k = {k}")
    # weight update
    for n in range(N):
        dz = Z[k] -h(px[n],Ld, l, Ll) # Measurement
        w[n] = PF_measurement_distribution.pdf(dz)# hint: PF_measurement_distribution.pdf
    w = w + eps # Some round off stuff, thx studass
    w = w / np.sum(w) # Normalize


    # resample
    i = 0
    noise = rng.random((1,1))/N
    cumweights = np.cumsum(w)
    u = np.zeros(N)
    for n in range(N):
        u = n/N + noise
        while u > cumweights[i]:
            i += 1
        # find a particle 'i' to pick
        # algorithm in the book, but there are other options as well
        pxn[n] = px[i]
    np.random.shuffle(pxn) # shuffle

    # trajecory sample prediction
    for n in range(n):
        # process noise, hint: PF_dynamic_distribution.rvs
        vkn = PF_dynamic_distribution.rvs() # just random vals drawn from dist
        px[n] = pendulum_dynamics_discrete(pxn[n], vkn, Ts, a) # particle prediction

    N_eff = 1 / np.sum(w**2)
    print(f"Degeneracy = {N_eff}")
    w = np.ones(N)/N # reset weigths
    
    # plot
    sch_particles.set_offsets(np.c_[l * np.sin(pxn[:, 0]), -l * np.cos(pxn[:, 0])])
    sch_true.set_offsets(np.c_[l * np.sin(x[k, 0]), -l * np.cos(x[k, 0])])

    fig4.canvas.draw_idle()
    plt.show(block=False)
    plt.waitforbuttonpress(plotpause)

plt.waitforbuttonpress()
