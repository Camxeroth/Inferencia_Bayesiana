"""Núcleo de inferencia bayesiana."""
from __future__ import annotations

from typing import Dict, List, Tuple
import scipy.stats as stats
import pymc as pm
import arviz as az


class BayesianUpdater:
    """Clase para actualizar analíticamente una distribución Beta-Binomial."""

    def __init__(self, alpha_0: float, beta_0: float) -> None:
        """Inicializa el modelo bayesiano con un prior Beta.

        Args:
            alpha_0 (float): Parámetro alfa inicial (> 0).
            beta_0 (float): Parámetro beta inicial (> 0).

        Raises:
            ValueError: Si los parámetros priors no son estrictamente positivos.
        """
        if alpha_0 <= 0.0 or beta_0 <= 0.0:
            raise ValueError("Los parámetros alpha_0 y beta_0 deben ser > 0")

        self.alpha = alpha_0
        self.beta = beta_0
        self.history: List[Tuple[float, float]] = [(self.alpha, self.beta)]

    def update(self, k: int, n: int) -> None:
        """Actualiza la distribución posterior dados los nuevos datos.

        Args:
            k (int): Número de éxitos en el lote (0 <= k <= n).
            n (int): Tamaño del lote (> 0).

        Raises:
            ValueError: Si k < 0, n <= 0, o k > n.
        """
        if n <= 0:
            raise ValueError("n debe ser mayor a 0")
        if k < 0 or k > n:
            raise ValueError("k debe estar entre 0 y n")

        self.alpha += k
        self.beta += n - k
        self.history.append((self.alpha, self.beta))

    def summary(self) -> Dict[str, float]:
        """Calcula estadísticas de resumen de la distribución posterior actual.

        Returns:
            Dict[str, float]: Diccionario con 'mean', 'median', 'variance', 'hdi_lower', 'hdi_upper'.
        """
        mean = self.alpha / (self.alpha + self.beta)
        median = stats.beta.ppf(0.5, self.alpha, self.beta)
        variance = (self.alpha * self.beta) / (
            ((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1)
        )
        
        # Extraemos muestras para arviz HDI
        samples = stats.beta.rvs(self.alpha, self.beta, size=10000)
        hdi = az.hdi(samples, prob=0.95)
        
        return {
            "mean": float(mean),
            "median": float(median),
            "variance": float(variance),
            "hdi_lower": float(hdi[0]),
            "hdi_upper": float(hdi[1])
        }

    def probability_above_threshold(self, threshold: float) -> float:
        """Calcula la probabilidad de que theta sea mayor a un umbral dado.

        Args:
            threshold (float): Valor umbral entre 0 y 1.

        Returns:
            float: Probabilidad P(θ > umbral) en la posterior actual.
            
        Raises:
            ValueError: Si threshold no está entre 0 y 1.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("El umbral debe estar entre 0 y 1")
        
        return 1.0 - stats.beta.cdf(threshold, self.alpha, self.beta)

    def run_mcmc_validation(self, k: int, n: int, draws: int = 2000) -> az.InferenceData:
        """Ejecuta un modelo MCMC en PyMC para validar la actualización analítica.

        Args:
            k (int): Éxitos acumulados totales.
            n (int): Intentos acumulados totales.
            draws (int, optional): Número de muestras MCMC. Defaults to 2000.

        Returns:
            az.InferenceData: Resultados de la inferencia con PyMC.
        """
        alpha_0, beta_0 = self.history[0]
        
        with pm.Model() as model:
            theta = pm.Beta("theta", alpha=alpha_0, beta=beta_0)
            pm.Binomial("y", n=n, p=theta, observed=k)
            trace = pm.sample(draws=draws, return_inferencedata=True, progressbar=False)
            
        return trace
