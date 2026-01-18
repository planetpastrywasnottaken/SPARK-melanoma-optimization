# SPARK: Simulation Platform for Adaptive Resistance Kinetics

This repository contains the core simulation and optimization code for the SPARK framework, an AI-assisted computational platform for optimizing melanoma treatment schedules targeting persister cells.

The model integrates:
- Mechanistic ODE-based signaling dynamics (MAPK and RTK/AKT)
- Bulk and persister tumor cell populations
- Pharmacokinetic and pharmacodynamic drug modeling
- Automated regimen optimization and evaluation

The code was developed as part of a research project submitted to the Presidential AI Challenge (Track II).

## Repository contents
- melanoma_model.py: Core ODE model and population dynamics
- Optimization scripts: Automated evaluation of treatment schedules
- params_literature.csv: Parameter values derived from published literature

## Notes
This code is provided for transparency and reproducibility. It is not intended for clinical use.
