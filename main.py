"""Orquestación del flujo de inferencia bayesiana."""
from __future__ import annotations

import os
import argparse
import arviz as az
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from data_simulator import DataStreamer
from bayesian_model import BayesianUpdater
from visualization import plot_prior_posterior, plot_hdi_evolution, animate_posterior_updates


def main():
    parser = argparse.ArgumentParser(description="Flujo de Inferencia Bayesiana: Modelo Beta-Binomial")
    parser.add_argument("--theta-true", type=float, default=0.7, help="Valor verdadero de theta")
    parser.add_argument("--alpha-0", type=float, default=1.0, help="Prior alpha inicial")
    parser.add_argument("--beta-0", type=float, default=1.0, help="Prior beta inicial")
    parser.add_argument("--threshold", type=float, default=0.6, help="Umbral para P(theta > umbral)")
    parser.add_argument("--n-batches", type=int, default=10, help="Número de lotes")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de cada lote")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    
    args = parser.parse_args()
    console = Console()
    
    try:
        streamer = DataStreamer(
            theta_true=args.theta_true,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            random_seed=args.seed
        )
        model = BayesianUpdater(alpha_0=args.alpha_0, beta_0=args.beta_0)
    except ValueError as e:
        console.print(f"[bold red]Error de configuración:[/bold red] {e}")
        return

    os.makedirs("outputs", exist_ok=True)
    
    table = Table(title="Actualización Bayesiana Lote por Lote")
    table.add_column("Lote", justify="right", style="cyan")
    table.add_column("Éxitos (k)", justify="right")
    table.add_column("Intentos (n)", justify="right")
    table.add_column("Media Post.", justify="right")
    table.add_column("HDI 95% Inferior", justify="right")
    table.add_column("HDI 95% Superior", justify="right")
    table.add_column(f"P(θ > {args.threshold})", justify="right", style="green")
    
    total_k = 0
    total_n = 0
    
    console.print("\n[bold]Iniciando simulación de datos...[/bold]")
    for i, (k, n) in enumerate(streamer.stream_batches(), 1):
        total_k += k
        total_n += n
        model.update(k, n)
        
        summary = model.summary()
        p_above = model.probability_above_threshold(args.threshold)
        
        table.add_row(
            str(i),
            str(k),
            str(n),
            f"{summary['mean']:.4f}",
            f"{summary['hdi_lower']:.4f}",
            f"{summary['hdi_upper']:.4f}",
            f"{p_above:.4f}"
        )
    
    console.print(table)
    
    console.print("\n[bold]Generando gráficos analíticos...[/bold]")
    # Plot Prior vs Posterior
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_prior_posterior(
        alpha_prior=args.alpha_0,
        beta_prior=args.beta_0,
        alpha_post=model.alpha,
        beta_post=model.beta,
        theta_true=args.theta_true,
        threshold=args.threshold,
        ax=ax
    )
    plt.savefig(os.path.join("outputs", "prior_posterior.png"))
    plt.close()
    
    # Plot HDI Evolution
    plot_hdi_evolution(model.history, args.theta_true)
    plt.savefig(os.path.join("outputs", "hdi_evolution.png"))
    plt.close()
    
    # Animate Updates
    animate_posterior_updates(
        model.history, 
        args.theta_true, 
        save_path=os.path.join("outputs", "posterior_update.gif")
    )
    console.print("[green]Gráficos guardados en el directorio 'outputs/'.[/green]")
    
    console.print("\n[bold]Validando modelo con MCMC (PyMC + NUTS)...[/bold]")
    trace = model.run_mcmc_validation(k=total_k, n=total_n, draws=2000)
    
    console.print("\n[bold]Resultados MCMC (ArviZ):[/bold]")
    mcmc_summary = az.summary(trace, var_names=["theta"])
    console.print(mcmc_summary.to_string())
    
    r_hat = mcmc_summary.loc["theta", "r_hat"]
    ess_bulk = mcmc_summary.loc["theta", "ess_bulk"]
    
    console.print("\n[bold]Diagnósticos de Convergencia:[/bold]")
    console.print(f"R-hat: {r_hat:.4f} (ideal cercano a 1.0)")
    console.print(f"ESS Bulk: {ess_bulk:.1f} (ideal > 100)")
    
    mcmc_mean = mcmc_summary.loc["theta", "mean"]
    analyt_mean = model.alpha / (model.alpha + model.beta)
    
    console.print("\n[bold]Comparación Analítica vs MCMC:[/bold]")
    console.print(f"Media Posterior Analítica: [cyan]{analyt_mean:.4f}[/cyan]")
    console.print(f"Media Posterior MCMC:      [cyan]{mcmc_mean:.4f}[/cyan]")
    console.print("\n[bold green]Proceso completado exitosamente.[/bold green]")


if __name__ == "__main__":
    main()
