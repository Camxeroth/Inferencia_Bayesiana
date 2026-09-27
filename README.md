<img width="1920" height="1080" alt="miniaturas japonesas" src="https://github.com/user-attachments/assets/96b31c78-f90f-4cd3-ab79-fa95c92a6e61" />

# Inferencia Bayesiana: Actualización Secuencial de Tasas de Conversión mediante el Modelo Conjugado Beta-Binomial

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)]()
[![License](https://img.shields.io/badge/license-MIT-informational)]()
[![Status](https://img.shields.io/badge/status-academic--prototype-lightgrey)]()

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Motivación y Contexto del Problema](#motivación-y-contexto-del-problema)
3. [Fundamento Estadístico](#fundamento-estadístico)
   - [Modelo Beta-Binomial Conjugado](#modelo-beta-binomial-conjugado)
   - [Intervalo de Alta Densidad (HDI)](#intervalo-de-alta-densidad-hdi)
   - [Validación mediante MCMC (NUTS)](#validación-mediante-mcmc-nuts)
   - [Probabilidad Posterior sobre un Umbral de Decisión](#probabilidad-posterior-sobre-un-umbral-de-decisión)
4. [Arquitectura del Repositorio](#arquitectura-del-repositorio)
5. [Instalación](#instalación)
6. [Uso de la Interfaz de Línea de Comandos](#uso-de-la-interfaz-de-línea-de-comandos)
7. [Parámetros de Configuración](#parámetros-de-configuración)
8. [Salidas del Sistema](#salidas-del-sistema)
9. [Pruebas Unitarias](#pruebas-unitarias)
10. [Supuestos y Limitaciones del Modelo](#supuestos-y-limitaciones-del-modelo)
11. [Trabajo Futuro](#trabajo-futuro)
12. [Referencias](#referencias)
13. [Licencia](#licencia)

---

## Resumen 

Este repositorio implementa un sistema de **inferencia bayesiana secuencial** para la estimación de una tasa de conversión (o de riesgo) binaria latente, denotada θ, a partir de un flujo de eventos observados en lotes. El núcleo analítico se basa en el modelo conjugado **Beta-Binomial**, lo que permite actualizar la distribución posterior de θ de forma cerrada (sin necesidad de métodos numéricos) a medida que llegan nuevos datos. Adicionalmente, se incorpora un módulo de validación basado en **Muestreo de Monte Carlo vía Cadenas de Markov (MCMC)**, utilizando el muestreador **NUTS** (No-U-Turn Sampler) implementado en `PyMC`, con el objetivo de contrastar empíricamente la solución analítica contra una aproximación numérica independiente.

El sistema está orientado a un caso de uso típico en analítica de producto y gestión de riesgo: la estimación temprana y con cuantificación de incertidumbre de una tasa de éxito binaria (conversión, adopción, incidencia de riesgo, tasa de defecto, etc.), donde la disponibilidad de datos es limitada y llega de forma incremental.

## Contexto del Problema

En escenarios de negocio y de ingeniería (pruebas A/B, control de calidad, detección temprana de riesgo operativo, tasas de adopción de nuevas funcionalidades), es común enfrentar la necesidad de estimar una proporción desconocida θ con **datos escasos y que llegan de forma incremental**. Los enfoques frecuentistas clásicos (estimación puntual vía máxima verosimilitud, intervalos de confianza asintóticos) presentan limitaciones relevantes en este contexto:

- Requieren tamaños de muestra considerables para que las aproximaciones asintóticas sean válidas.
- No incorporan de manera natural conocimiento previo (*prior knowledge*) sobre el fenómeno.
- No proveen una interpretación probabilística directa sobre el parámetro de interés (un intervalo de confianza no equivale a "la probabilidad de que θ esté en este rango").

El enfoque bayesiano resuelve estas limitaciones de forma natural: permite incorporar conocimiento previo mediante una distribución *prior*, actualizarlo de forma exacta y secuencial conforme llegan los datos, y obtener enunciados probabilísticos directos sobre θ (por ejemplo, "la probabilidad de que la tasa de conversión supere el 15% es 0.87"), los cuales son directamente accionables en un contexto de toma de decisiones.

## Fundamento Estadístico

### Modelo Beta-Binomial Conjugado

Sea θ ∈ (0, 1) la tasa de conversión (o de riesgo) latente que se desea estimar. Se asume el siguiente modelo generativo:

**Distribución previa (*prior*):**

θ ~ Beta(α₀, β₀)

**Función de verosimilitud (*likelihood*):**

y | θ ~ Binomial(n, θ)

donde `n` es el número de ensayos observados en un lote y `y = k` es el número de éxitos.

**Distribución posterior:**

Dada la propiedad de conjugación entre la familia Beta y la familia Binomial, la distribución posterior tiene forma cerrada:

θ | k, n ~ Beta(α₀ + k, β₀ + n − k)

Esta propiedad es la que permite la **actualización secuencial exacta**: al llegar un nuevo lote de datos, los parámetros posteriores del lote anterior se emplean como los nuevos parámetros *prior*, sin pérdida de información y sin necesidad de recomputar sobre el histórico completo de observaciones:

α_t = α_{t−1} + k_t
β_t = β_{t−1} + (n_t − k_t)

### Intervalo de Alta Densidad (HDI)

El **Intervalo de Alta Densidad (HDI)** al (1 − a) de credibilidad se define como el intervalo más estrecho [L, U] tal que:

P(L ≤ θ ≤ U | datos) = 1 − a

y todo punto dentro del intervalo tiene una densidad posterior mayor o igual que cualquier punto fuera de él. A diferencia de un intervalo de confianza frecuentista, el HDI admite una interpretación probabilística directa: *"existe un 95% de probabilidad posterior de que θ se encuentre en este intervalo, dados los datos observados y el prior asumido"*.

### Validación mediante MCMC (NUTS)

Si bien el modelo Beta-Binomial admite solución analítica exacta, el repositorio incorpora un módulo de validación independiente basado en **Muestreo de Monte Carlo vía Cadenas de Markov**, específicamente el algoritmo **NUTS** (*No-U-Turn Sampler*), implementado sobre `PyMC`. El propósito de este módulo es doble:

1. **Verificación cruzada:** confirmar que la solución analítica y la solución numérica convergen al mismo resultado, sirviendo como control de calidad del pipeline.
2. **Extensibilidad metodológica:** sentar las bases para migrar el modelo hacia especificaciones más complejas (jerárquicas, con covariables, con no estacionariedad) que ya no admitan solución conjugada cerrada y requieran inferencia aproximada por defecto.

El diagnóstico de convergencia de las cadenas se reporta mediante los estadísticos estándar de la literatura: **R-hat** (Gelman-Rubin, valores cercanos a 1.0 indican convergencia) y **ESS** (*Effective Sample Size*, tamaño de muestra efectivo tras corregir por autocorrelación).

### Probabilidad Posterior sobre un Umbral de Decisión

Dado un umbral de negocio τ (por ejemplo, τ = 0.15), el sistema calcula:

P(θ > τ | datos) = 1 − F_Beta(τ; α_n, β_n)

donde F_Beta es la función de distribución acumulada de la Beta posterior. Esta cantidad constituye el insumo directo para la toma de decisiones bajo incertidumbre (por ejemplo, activar una alerta de riesgo o declarar ganador un experimento cuando dicha probabilidad supera un umbral de confianza operativo).

## Arquitectura del Repositorio

```
Inferencia_Bayesiana/
├── main.py                  # Orquestación de la CLI y del flujo de ejecución end-to-end
├── bayesian_model.py         # Clase BayesianUpdater: actualización conjugada y validación MCMC
├── data_simulator.py         # Clase DataStreamer: simulación del flujo de eventos binarios
├── visualization.py          # Funciones de graficación (prior/posterior, HDI, evolución)
├── tests/                    # Suite de pruebas unitarias (pytest)
├── outputs/                  # Artefactos generados: gráficos y resúmenes por ejecución
├── requirements.txt           # Especificación de dependencias y versiones
└── README.md                  # Documentación del proyecto
```

**Separación de responsabilidades:**

| Módulo | Responsabilidad | Principales dependencias |
|---|---|---|
| `data_simulator.py` | Generación reproducible de flujos de datos binarios (`Binomial(n, θ_true)`) | `numpy` |
| `bayesian_model.py` | Actualización conjugada, cálculo de estadísticos posteriores, validación MCMC | `scipy`, `pymc`, `arviz` |
| `visualization.py` | Representación gráfica de la evolución prior → posterior y del HDI | `matplotlib`, `seaborn` |
| `main.py` | Orquestación del flujo completo vía CLI (`argparse`) | — |

## Instalación

**Requisitos previos:** Python ≥ 3.9.

```bash
# 1. Clonar el repositorio
git clone https://github.com/Camxeroth/Inferencia_Bayesiana.git
cd Inferencia_Bayesiana

# 2. Crear y activar un entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota de compatibilidad:** las versiones recientes de `arviz` / `arviz-stats` (≥ 1.0) renombraron el argumento `hdi_prob` a `prob` en la función `az.hdi()`. Si se utiliza una versión de `arviz` posterior a la especificada en `requirements.txt`, verificar la firma de dicha función antes de ejecutar, para evitar errores del tipo `TypeError: hdi got an unexpected keyword argument`.

## Uso de la Interfaz de Línea de Comandos

**Ejecución con configuración por defecto:**

```bash
python main.py
```

**Ejecución con parámetros personalizados:**

```bash
python main.py --theta-true 0.85 --alpha-0 2.0 --beta-0 2.0 --batch-size 100 --n-batches 5 --threshold 0.80
```

Durante la ejecución, el sistema imprime en consola un resumen estadístico por cada lote procesado (media posterior, mediana, HDI 95%, probabilidad posterior sobre el umbral) y almacena los artefactos gráficos generados en el directorio `outputs/`.

## Parámetros de Configuración

| Parámetro | Descripción | Tipo |
|---|---|---|
| `--theta-true` | Tasa de conversión/riesgo real utilizada por el simulador para generar los datos (solo disponible en modo de simulación experimental) | `float` |
| `--alpha-0` | Parámetro α del prior Beta | `float` |
| `--beta-0` | Parámetro β del prior Beta | `float` |
| `--batch-size` | Número de ensayos (`n`) por lote simulado | `int` |
| `--n-batches` | Número total de lotes a procesar secuencialmente | `int` |
| `--threshold` | Umbral de decisión τ sobre el cual se calcula P(θ > τ \| datos) | `float` |

> El listado completo y actualizado de flags disponibles, junto con sus valores por defecto exactos, puede consultarse en cualquier momento mediante:
>
> ```bash
> python main.py --help
> ```

## Salidas del Sistema

- **Consola:** tabla de resumen estadístico por iteración (media, mediana, HDI 95%, P(θ > τ)), y comparación entre la solución analítica y la validación MCMC (media posterior, R-hat, ESS).
- **`outputs/`:** gráficos de la evolución de la distribución posterior a lo largo de los lotes, con marcación del HDI 95% y de la tasa real θ_true utilizada por el simulador (esta última visible únicamente en el contexto de validación experimental, no observable en un escenario real de producción).

## Pruebas Unitarias

La lógica de actualización conjugada y de cálculo probabilístico está cubierta mediante `pytest`:

```bash
pytest tests/
```

Se recomienda ejecutar la suite de pruebas tras cualquier modificación al núcleo de inferencia (`bayesian_model.py`), dado que constituye la garantía de corrección del componente analítico del sistema.

## Supuestos y Limitaciones del Modelo

1. **Independencia e idéntica distribución (i.i.d.):** se asume que cada evento observado es condicionalmente independiente dado θ, y que todos los eventos comparten la misma probabilidad subyacente.
2. **Estacionariedad de θ:** el modelo asume que la tasa verdadera no varía en el tiempo. En escenarios de *concept drift* (cambio de comportamiento del fenómeno subyacente), esta suposición puede resultar inadecuada y conducir a estimaciones desactualizadas si no se introduce un mecanismo de olvido (*discounting*) sobre el prior.
3. **Especificación del prior:** la elección de (α₀, β₀) incide directamente en la velocidad de convergencia de la posterior, particularmente relevante cuando el tamaño muestral acumulado es reducido.
4. **Ausencia de covariables:** el modelo actual es univariante y no incorpora heterogeneidad explicada por variables explicativas (segmentación por canal, cohorte, geografía, etc.).

##  Futuro

- Incorporación de un factor de olvido (*exponential forgetting*) sobre los parámetros del prior para escenarios no estacionarios.
- Extensión a un modelo jerárquico Beta-Binomial para estimación simultánea de múltiples grupos o segmentos con *partial pooling*.
- Incorporación de covariables mediante una especificación de regresión logística bayesiana como alternativa al modelo conjugado puro.
- Automatización de la comparación analítica vs. MCMC como prueba de regresión continua (CI/CD).

## Referencias

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
- Kruschke, J. K. (2015). *Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan* (2nd ed.). Academic Press.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.
- Documentación oficial de PyMC: https://www.pymc.io/
- Documentación oficial de ArviZ: https://python.arviz.org/

## Licencia

Este proyecto se distribuye bajo los términos que el autor determine en el archivo `LICENSE` del repositorio. En ausencia de dicho archivo, se recomienda incorporar explícitamente una licencia (por ejemplo, MIT) para clarificar los términos de uso, modificación y distribución.
