# WinnerTakeAll
Builds 2-particle correlator distributions for proton-proton collisions, comparing winner-take-all and standard jet axes.


Overview of analysis pipeline:
1. On cluster:
    1. Make the signals, then merge them
    2. Make the backgrounds, then merge them
    3. copy results to local
2. Locally:
    1. Make the yields out of the signals and backgrounds
    2. Extract and compare V2 for the different data and different cluster methods


key plots:
- *v2 versus jet multiplicity in WTA frame and standard frame*: output/new_radius/comparing_frames/stitched/combined_coeff_v2_summary_plot.pdf