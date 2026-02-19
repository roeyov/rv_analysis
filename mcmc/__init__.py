"""
mcmc — Markov Chain Monte Carlo posterior sampling for orbital parameters.

Submodules:
    models          Log-prior/likelihood/probability for ecc/circ/null models
    runner          emcee MCMC wrappers (run_mcmc_ecc/circ/null)
    analysis        Lucy-Sweeney test, chain summary statistics
    mcmc_plotting   Corner plots, phase-folded orbit bands
    batch           Batch MCMC runner over lmfit results
    selector_app    Streamlit best-row chooser comparison
"""
