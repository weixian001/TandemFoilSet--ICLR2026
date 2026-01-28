# CFD Dataset for TandemFoilSet ICLR2026

This repository contains the dataset generation and extraction code for high-fidelity CFD simulations used in our flow prediction benchmarks.
It supports a wide range of aerodynamic configurations, from single and tandem airfoils to race cars.

# CFD Dataset Generation Codes (GenData)

This repository contains dataset generation pipelines for computational fluid dynamics (CFD) simulations across three different geometry families:

1. **Tandem Cruise Fixed** :
        tandemAirfoil/autoSingle_woO_match: to generate single-airfoil datasets at AOA=0 and 5
        tandemAirfoil/tandem_cruise: to generate tandem-airfoil curise datasets at AOA=0 and 5
        tandemAirfoil/tandem_takeoff: to generate tandem-airfoil takeoff datasets at AOA=5 with ground effect
2. **Random Fields** :
        randomFields/run_random_single: to generate single-airfoil datasets at random flow field
        randomFields/run_random_cruise: to generate tandem-airfoil datasets at random flow field
3. **Race Cars** :
        raceCar/run_singleRaceCar: to generate inverted single-airfoil datasets with ground effect at random flow field
        raceCar/run_raceCar: to generate race car datasets with gorund effect at random flow field

Each subfolder includes scripts to create geometry, mesh it, and run simulations.

# OpenFOAM CFD Data Extraction Codes (ExtracData)



## Dataset Access

- **Single-Airfoil Datasets** (pickle)
  [🔗 NTU Dataverse](https://doi.org/10.21979/N9/9OYSTD)

- **Tandem-Airfoil Datasets** (raw + pickle)
  [🔗 NTU Dataverse](To be updated)
