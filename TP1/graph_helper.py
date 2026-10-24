import numpy as np
import matplotlib.pyplot as plt


def plot_with_interval(
    x,
    curves,
    lower=None,
    upper=None,
    interval_label=None,
    interval_color="lightgray",
    interval_alpha=0.7,
    xlabel=None,
    ylabel=None,
    title=None,
    grid=True,
    ax=None
):
    """
    Affiche plusieurs courbes et, éventuellement, un intervalle [lower, upper].

    Parameters
    ----------
    x : array_like, shape (N,)
        Abscisses.

    curves : list of dict
        Chaque dictionnaire décrit une courbe :
            {
                "y": vecteur de taille N,
                "label": "nom",
                "color": "blue",       # optionnel
                "linestyle": "-",      # optionnel
                "linewidth": 1.5       # optionnel
            }

    lower, upper : array_like, shape (N,), optional
        Bornes inférieure et supérieure de la zone remplie.

    interval_label : str, optional
        Nom de la zone dans la légende.

    interval_color : couleur matplotlib
        Couleur de la zone.

    interval_alpha : float
        Transparence de la zone.

    ax : matplotlib.axes.Axes, optional
        Axe existant. Si None, un nouvel axe est créé.
    """

    x = np.asarray(x)

    if ax is None:
        fig, ax = plt.subplots()

    # Intervalle d'incertitude
    if lower is not None and upper is not None:
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        ax.fill_between(
            x,
            lower,
            upper,
            color=interval_color,
            alpha=interval_alpha,
            label=interval_label
        )

    # Courbes
    for curve in curves:
        ax.plot(
            x,
            curve["y"],
            label=curve.get("label", None),
            color=curve.get("color", None),
            linestyle=curve.get("linestyle", "-"),
            linewidth=curve.get("linewidth", 1.5)
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.legend()

    return ax

if False:
    t = np.arange(10)

    x_true = np.array([20, 30, 43, 55, 68, 80, 91, 103, 114, 126])
    x_est  = np.array([0, 29, 42, 56, 67, 79, 92, 102, 115, 125])

    sigma = np.array([10, 4, 3, 2.5, 2.2, 2, 1.8, 1.7, 1.6, 1.5])

    lower = x_est - 3 * sigma
    upper = x_est + 3 * sigma

    plot_with_interval(
        t,

        curves=[
            {
                "y": x_true,
                "label": "position réelle",
                "color": "blue"
            },
            {
                "y": x_est,
                "label": "position estimée",
                "color": "red"
            }
        ],

        lower=lower,
        upper=upper,
        interval_label=r"incertitude ($3\sigma$)",

        xlabel="temps (s)",
        ylabel="position (m)"
    )

    plt.show()
