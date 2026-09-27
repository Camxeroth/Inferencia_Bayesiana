"""Funciones de visualización para el modelo bayesiano."""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def plot_prior_posterior(
    alpha_prior: float,
    beta_prior: float,
    alpha_post: float,
    beta_post: float,
    theta_true: float,
    threshold: float,
    ax: Optional[plt.Axes] = None
) -> None:
    """Grafica la distribución Prior y Posterior junto con el valor verdadero y umbral.

    Args:
        alpha_prior (float): Alfa del prior.
        beta_prior (float): Beta del prior.
        alpha_post (float): Alfa del posterior.
        beta_post (float): Beta del posterior.
        theta_true (float): Valor verdadero de theta.
        threshold (float): Umbral de interés.
        ax (Optional[plt.Axes], optional): Eje matplotlib donde graficar. Defaults to None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(0, 1, 500)
    y_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
    y_post = stats.beta.pdf(x, alpha_post, beta_post)

    sns.lineplot(x=x, y=y_prior, color="blue", linestyle="--", label="Prior", ax=ax)
    sns.lineplot(x=x, y=y_post, color="red", label="Posterior", ax=ax)
    
    ax.axvline(theta_true, color="green", linestyle="-", label=r"$\theta_{true}$")
    ax.axvline(threshold, color="orange", linestyle=":", label="Threshold")
    
    ax.set_title("Prior vs Posterior")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Density")
    ax.legend()


def plot_hdi_evolution(history: List[Tuple[float, float]], theta_true: float) -> None:
    """Grafica la evolución del HDI 95% y la media posterior por iteración.

    Args:
        history (List[Tuple[float, float]]): Historial de parámetros (alpha, beta).
        theta_true (float): Valor verdadero de theta.
    """
    means = []
    hdi_lowers = []
    hdi_uppers = []
    import arviz as az
    
    for a, b in history:
        means.append(a / (a + b))
        samples = stats.beta.rvs(a, b, size=5000, random_state=42)
        hdi_interval = az.hdi(samples, prob=0.95)
        hdi_lowers.append(hdi_interval[0])
        hdi_uppers.append(hdi_interval[1])

    iterations = range(len(history))
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=iterations, y=means, color="blue", label="Posterior Mean")
    plt.fill_between(
        iterations, hdi_lowers, hdi_uppers, color="blue", alpha=0.2, label="95% HDI"
    )
    plt.axhline(theta_true, color="green", linestyle="--", label=r"$\theta_{true}$")
    
    plt.title("Evolution of Posterior Mean and 95% HDI")
    plt.xlabel("Iteration / Batch")
    plt.ylabel(r"$\theta$")
    plt.legend()
    plt.tight_layout()


def animate_posterior_updates(
    history: List[Tuple[float, float]],
    theta_true: float,
    save_path: str | None = None
) -> None:
    """Crea y guarda una animación de la actualización de la distribución posterior.

    Args:
        history (List[Tuple[float, float]]): Historial de parámetros (alpha, beta).
        theta_true (float): Valor verdadero de theta.
        save_path (str | None, optional): Ruta donde guardar la animación. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 1, 500)
    
    line, = ax.plot([], [], lw=2, color='red', label="Posterior")
    ax.axvline(theta_true, color="green", linestyle="--", label=r"$\theta_{true}$")
    
    ax.set_xlim(0, 1)
    max_alpha, max_beta = history[-1]
    # Calculamos un límite superior estimado del eje y analizando pdf del prior y post final
    # Para evitar warnings de inf, acotamos
    max_y = min(max(stats.beta.pdf(x, max_alpha, max_beta)) * 1.1, 50.0) 
    ax.set_ylim(0, max_y)
    
    ax.set_title("Posterior Update Over Time")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        a, b = history[frame]
        y = stats.beta.pdf(x, a, b)
        line.set_data(x, y)
        ax.set_title(f"Posterior Update Over Time (Batch: {frame})")
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), init_func=init, blit=True, repeat=False
    )
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=2)
    else:
        plt.show()
