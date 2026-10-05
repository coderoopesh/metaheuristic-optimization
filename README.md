## Citation
* **Source Code DOI:** https://doi.org/10.5281/zenodo.23151479

If you use this in your research, please cite the source code artifact.

---

# Metaheuristic TSP Solver

A C++ implementation of a hybrid metaheuristic solver for the Traveling Salesman Problem (TSP), optimized for large-scale TSPLIB benchmark instances on Linux environments.

---

## Features

- **Hybrid Metaheuristic:** Integrates Variable Neighborhood Search (VNS) and Large Neighborhood Search (LNS) principles.
- **Speed Optimizations:** Utilizes Don't Look Bits (DLB) to skip inactive nodes during local search and dynamic stage-based perturbation thresholds.

---

## System Requirements

- **Compiler:** GCC supporting C++17 or higher (`g++`).
- **OS:** Linux (Tested and optimized on Fedora Linux).

---

## Compilation

```bash
g++ -O3 -march=native -flto -DNDEBUG -std=c++17 -Wall -Wextra -pipe tsp_solver.cpp -o tsp_solver
