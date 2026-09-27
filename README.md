# Inferencia Bayesiana: Modelo Beta-Binomial

Este proyecto provee una implementación modular de un modelo bayesiano de actualización conjugada Beta-Binomial. Incluye la simulación de un flujo de datos (streaming), actualización analítica paso a paso, visualización de las distribuciones, cálculo de Intervalos de Alta Densidad (HDI), y la validación cruzada con una implementación MCMC usando PyMC.

## Fundamento Teórico

### Modelo Beta-Binomial Conjugado
La distribución Beta es conjugada respecto a la verosimilitud Binomial.
- **Prior:** $\theta \sim \text{Beta}(\alpha_0, \beta_0)$
- **Likelihood:** $y \sim \text{Binomial}(n, \theta)$
- **Posterior:** Después de observar $k$ éxitos en $n$ ensayos, la distribución posterior se calcula de forma analítica como:
  $\theta | k, n \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

### HDI (Highest Density Interval)
El HDI indica el intervalo más estrecho que contiene una probabilidad determinada (por ejemplo, 95% de la curva). Garantiza que todos los puntos dentro del intervalo tienen mayor densidad de credibilidad marginal respecto a a los de fuera, útil para inferencia.

### Validación Continua mediante MCMC (NUTS)
El núcleo en Python ha sido acompañado de un módulo en **PyMC** para proveer validación sobre los mismos datos. Correr NUTS nos permite comparar la esperanza o *media posterior* deducida por la matemática pura comparada contra el muestreo log-verosímil.

### Limitaciones y Supuestos del Modelo
- **Independencia de Eventos:** El modelo asume eventos independientes distribuidos idénticamente (i.i.d).
- **Estacionariedad de $\theta$:** Asumimos que la tasa de conversión global no cambia. En escenarios dinámicos del mundo real, una "pérdida de memoria" o factor de olvido podría ser deseable en contra del Prior estático.

---

## Instrucciones de Instalación

1. Asegúrate de tener Python (versión >= 3.9 recomendada).
2. Clona o en este caso accede al repositorio del directorio local.
3. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   # En macOS / Linux
   source venv/bin/activate
   ```
4. Instala todas las librerías del proyecto:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución del Sistema CLI

La interfaz de línea de comandos está implementada con `argparse` y `rich` para mostrar gráficamente las tablas directamente en la terminal.

**Uso general / Por Defecto:**
```bash
python main.py
```

**Ejemplo Personalizado:**
Puedes proveer explícitamente priors débiles o fuertes, así como modular el *streaming* del lote de datos.
```bash
python main.py --theta-true 0.85 --alpha-0 2.0 --beta-0 2.0 --batch-size 100 --n-batches 5 --threshold 0.80
```

## Pruebas Unitarias
El componente de aserción probabilística y lógica está testeado enteramente con **pytest**. Para validar los conjuntos de forma independiente, ejecuta:

```bash
pytest tests/
```
