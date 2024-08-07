# Membrane Process Modeling - Techno-Economic Analysis
(meprom-tea)

by Aron K. Beke
King Abdullah University of Science and Technology

This repository contains code to support the manuscript "Tipping the chemical separation iceberg: hybrid modeling for nanofiltration in energy-conscious manufacturing".
(DOI:XX/XYZ)

Two functionalities are included, both run from command line:
1. Running industrial_example_reproduction.py will perform calculations and display results related to two industrial example separation steps included in the manuscript.
2. Running main.py allows the user to perform simple techno-economic analysis supporting technology selection between nanofiltration, evaporation, and extraction.

# main.py
Running main.py will prompt the user for the name or path of a .txt file containing all necessary separation parameters. The input.txt file is provided as template.
After performing the necessary calculations, the code will output the following:

1. Best technology: energetically most favorable technology or combination of technologies
2. Specific energy demand of best technology in J / kg solute
3. Specific energy demand of conventional technology (evaporation or extraction-evaporation) in J / kg solute
4. Specific energy demand reduction
5. Reduction in CO2 equivalent emissions (global warming potential)
6. Reduction in operating costs (including membrane consumption)
7. Threshold membrane price: below this membrane price, the best technology becomes economically favorable (as far as operating costs are concerned)

Please provide the necessary input parameters according to the following considerations:
- Separation type: "solute_concentration", "solvent_recovery", "solute_separation", or "impurity_removal"
- Concentration data have to be in kg / m3
- Solvent has to be one of the solvents listed in data/solvent_properties
- Solute target ratio is understood as solute1 target concentration divided by solute2 target concentration
- Solute rejection values have to be between -0.5 and 1.
- Density in kg / m3
- Water partitioning coefficients are understood as concentration in water / concentration in solvent. The log10 of this value is required.
- Heptane partitioning coefficients are understood as concentration in heptane / concentration in solvent. The log10 of this value is required.
- External heat integration has to be between 0 and 1.
- Solvent permeance has to be in LMHbar (L / m2 / h / bar)
- Membrane cost in USD / m2
- Membrane lifetime in years
- Country has to be one of the countries listed in data/technoeconomic_data.csv

Nanofiltration is simulated at 20 barg transmembrane pressure. Solute permeance is inferred from rejection at 20 barg.
In ternary separation simulations the provided feed concentrations might be reduced (dilution) to ensure compliance with solubility limitations. Ratio of feed concentrations is not altered.