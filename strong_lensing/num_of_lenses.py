import numpy as np
from scipy.constants import c
from scipy.integrate import quad
from styles.matplotlib_style import *
import matplotlib as mpl
from matplotlib.colors import LogNorm

plot = True


def d_com(z, Omega_m, H_0):
    # H_0 in km/s/Mpc.
    d_H = c/(H_0*3.241e-20)  # 1km/s/Mpc equals to 3.241×10^-20 per second
    Omega_lambda = 1-Omega_m

    # Define the integrand as a function of a single variable
    def integrand(z_prime):
        return 1 / np.sqrt(Omega_m * (1 + z_prime) ** 3 + Omega_lambda)

    # Integrate from 0 to z
    integral = quad(integrand, 0, z)
    return d_H * integral[0]


def D_A(z, comoving_distance):
    # comoving_distance is the value of the comoving distance at the single value z
    return 1/(1+z) * comoving_distance


def d_proper(z, Omega_m, H_0):
    return d_com(z, Omega_m, H_0)


def N(z, Omega_m, H_0):
    # Now, z must be a numpy array
    # This returns N \propto (1+z)^3 * D^2(z) * d/dz d_proper := A * B * C
    d_comoving = np.array([d_com(redshift, Omega_m, H_0) for redshift in z])
    A = (1+z)**3
    B = np.array([D_A(redshift, d_comoving[idx])**2 for idx, redshift in enumerate(z)])
    C = np.gradient(d_comoving, z)

    res = A * B * C
    # norm the result for visual purposes:
    res /= 1e81
    return res


num_evals = 100
redshifts = np.linspace(0, 2.5, num_evals)
Omega_ms = np.linspace(0, 1, num_evals)
H_0s = np.linspace(10, 120, num_evals)
if plot:

    fig, axs = plt.subplots(1, 2)
    colors0 = mpl.colormaps["plasma"](np.linspace(0, 1, num_evals))  # Colormap
    colors1 = mpl.colormaps["plasma"](np.linspace(0, 1, num_evals))  # Colormap


    for Omega_m, color in zip(Omega_ms, colors0):
        axs[0].plot(redshifts, N(redshifts, Omega_m=Omega_m, H_0=70), color=color, ls="-", markersize=0)

    for H_0, color in zip(H_0s, colors1):
        axs[1].plot(redshifts, N(redshifts, Omega_m=0.3, H_0=H_0), color=color, ls="-", markersize=0)

    sm0 = plt.cm.ScalarMappable(cmap=mpl.colormaps["plasma"], norm=plt.Normalize(vmin=min(Omega_ms), vmax=max(Omega_ms)), )
    sm1 = plt.cm.ScalarMappable(cmap=mpl.colormaps["plasma"], norm=plt.Normalize(vmin=min(H_0s), vmax=max(H_0s)), )
    sm0.set_array([])
    sm1.set_array([])
    cbar0 = plt.colorbar(sm0, ax=axs[0])
    # cbar0.set_label(r'$\Omega_m$', labelpad=10, loc='top')
    cbar1 = plt.colorbar(sm1, ax=axs[1])
    # cbar1.set_label(r'$H_0$', labelpad=10, loc='top')

    axs[0].text(0.265, 0.04783, r"$H_0=70\mathrm{km/s/Mpc}$", fontsize=10)
    axs[1].text(0.265, 1.320, r"$\Omega_m=0.3$", fontsize=10)
    axs[0].set_xlabel("$z$")
    axs[1].set_xlabel("$z$")
    axs[0].set_ylabel("$N(z)$")
    axs[0].set_title(r"Varying $\Omega_m$ from $0$ to $1$", loc="left", fontsize=12)
    axs[1].set_title(r"Varying $H_0$ from $10$ to $120\mathrm{km/(s\cdot Mpc)}$", loc="right", fontsize=12)
    fig.tight_layout()
    plt.savefig("figures/T7_i.png")
    plt.show()

num_evals_grid_search = 30
# Overwriting old variables
Omega_ms = np.linspace(0, 0.3, num_evals_grid_search)
H_0s = np.linspace(1, 30, num_evals_grid_search)

results = np.zeros((num_evals_grid_search, num_evals_grid_search))
for i in range(num_evals_grid_search):
    for j in range(num_evals_grid_search):
        Omega_m = Omega_ms[i]
        H_0 = H_0s[j]
        # taking the mean as a measure
        results[i, j] = np.mean(N(z=redshifts, Omega_m=Omega_m, H_0=H_0))

results /= np.sum(results)
max_idx = np.where(results == np.max(results))
maximum = results[max_idx]

X, Y = np.meshgrid(Omega_ms[range(num_evals_grid_search)], H_0s[range(num_evals_grid_search)])
plt.contourf(X, Y, results, levels=100, norm=LogNorm())
# Plot the arrow
prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0)
plt.annotate("", xytext=(0.0634, 7.37), xy=(Omega_ms[max_idx[0]][0], H_0s[max_idx[1]][0]), arrowprops=prop)
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$H_0$ in $\mathrm{km(s\cdot Mpc)}$")
plt.colorbar()
plt.savefig("figures/T7_ii.png")
plt.show()

