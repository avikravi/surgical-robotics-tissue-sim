# Surgical Robotics Tissue Simulation

A Genesis-based dataset generation pipeline for surgical tissue material characterization.
Modeled after the MASIV dataset.

## Project Structure

- data/simulations/ — Per-simulation folders (objectID_sequenceID format)
- data/videos/ — Multi-view videos per simulation
- data/gifs/ — GIFs for visualization
- src/simulate/ — Genesis simulation scripts
- src/dataset/ — Data saving and metadata utilities
- src/cameras/ — Camera setup and multi-view logic
- notebooks/ — Jupyter notebooks for experimentation
- configs/ — Simulation config files

## Research Context

Summer research project on tissue material characterization from RGB video during robotic surgery.
Related work: DiSECt, GIC, PAC-NeRF, MASIV.
