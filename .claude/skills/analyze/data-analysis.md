# Data Analysis Skill

Use this skill when analysing simulation data, plotting results, or working with PyDASA outputs.

## Rules

- Read notebook markdown cells for context before analysing data
- Verify dictionary key formats and variable names before processing
- Use intermediate variables for readability in data transformations
- Do not chain pandas operations excessively

## Plotting Conventions

- Publication-quality figures (clear labels, proper font sizes)
- Use consistent colour schemes across related plots
- Annotate key features (e.g., K families, c configurations)
- Support both 2D projections and 3D scatter plots for Yoly Charts

## PACS Simulation Context

- Iteration 1: Single-node M/M/c/K, 27 configurations (3x3x3)
- Iteration 2: 7-node Jackson network, 25,200 experiments, 5 routing scenarios
- Four primary DCs: theta (Occupancy), sigma (Stall), eta (Effective Yield), phi (Memory Efficiency)
- Network-level DC: sigma_e2e (End-to-End Stall)
- Routing scenarios: 100R, 80R20W, 50R50W, 20R80W, 100W
