"""Simulación de flujo de datos."""
from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np


class DataStreamer:
    """Clase para simular un flujo de datos binomiales."""

    def __init__(
        self,
        theta_true: float,
        batch_size: int,
        n_batches: int,
        random_seed: int | None = None,
    ) -> None:
        """Inicializa el simulador de datos.

        Args:
            theta_true (float): Probabilidad verdadera de éxito (0 < theta_true < 1).
            batch_size (int): Tamaño de cada lote de datos (> 0).
            n_batches (int): Número de lotes a generar (> 0).
            random_seed (int | None, optional): Semilla para reproducibilidad. Defaults to None.

        Raises:
            ValueError: Si los parámetros no están en los rangos esperados.
        """
        if not (0.0 < theta_true < 1.0):
            raise ValueError("theta_true debe ser estrictamente entre 0 y 1")
        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor a 0")
        if n_batches <= 0:
            raise ValueError("n_batches debe ser mayor a 0")

        self.theta_true = theta_true
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.rng = np.random.default_rng(random_seed)

    def stream_batches(self) -> Iterator[Tuple[int, int]]:
        """Genera lotes de datos simulados de forma secuencial.

        Yields:
            Iterator[Tuple[int, int]]: Tupla (k, n) con el número de éxitos k y el tamaño del lote n.
        """
        for _ in range(self.n_batches):
            k = int(self.rng.binomial(n=self.batch_size, p=self.theta_true))
            yield k, self.batch_size
