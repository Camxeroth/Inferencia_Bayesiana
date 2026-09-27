"""Pruebas unitarias para el modelo bayesiano."""
import pytest
from bayesian_model import BayesianUpdater

def test_initialization():
    """Prueba la inicialización con parámetros válidos e inválidos."""
    model = BayesianUpdater(1.0, 1.0)
    assert model.alpha == 1.0
    assert model.beta == 1.0
    
    with pytest.raises(ValueError):
        BayesianUpdater(-1.0, 1.0)

def test_analytic_update():
    """Prueba la actualización analítica conjugada de la distribución."""
    model = BayesianUpdater(1.0, 1.0)
    model.update(k=7, n=10)
    assert model.alpha == 8.0
    assert model.beta == 4.0
    assert len(model.history) == 2
    
def test_update_validation():
    """Prueba que el método update levante excepciones ante valores incoherentes."""
    model = BayesianUpdater(1.0, 1.0)
    with pytest.raises(ValueError):
        model.update(k=-1, n=5)
    with pytest.raises(ValueError):
        model.update(k=6, n=5)
    with pytest.raises(ValueError):
        model.update(k=5, n=0)

def test_probability_above_threshold():
    """Prueba que la probabilidad decrece monotónicamente al aumentar el umbral."""
    model = BayesianUpdater(10.0, 10.0)  # Media asintótica de 0.5
    prob_0_3 = model.probability_above_threshold(0.3)
    prob_0_5 = model.probability_above_threshold(0.5)
    prob_0_7 = model.probability_above_threshold(0.7)
    
    assert prob_0_3 > prob_0_5
    assert prob_0_5 > prob_0_7

def test_probability_threshold_bounds():
    """Prueba los límites prohibidos explícitamente en el umbral."""
    model = BayesianUpdater(10.0, 10.0)
    with pytest.raises(ValueError):
        model.probability_above_threshold(-0.1)
    with pytest.raises(ValueError):
        model.probability_above_threshold(1.1)
