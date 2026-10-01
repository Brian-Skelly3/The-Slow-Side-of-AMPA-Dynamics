# The-Slow-Side-of-AMPA-Dynamics

Accompanying code and data for research paper, "The slow side of AMPA Dynamics: A Parsimoneous framwork for TARP-dependant receptor dynamics" https://dx.doi.org/10.21203/rs.3.rs-9972009/v1

The folder, "Data," contains all of the data necessary for this paper. Each set of data can be simulated in the corresponding python files by uncommenting out the relevant np.savetxt function. Example; to find the errors of the different simulations, go to Creating_Data.ipynb -> Errors -> Uncomment the necessary save cell -> Run the corrresponding cell.

"Creating_Data" explains how the synaptic models are defined and utilised. This file provides all of the simulations except for network simulations and optimisations.

"1neur1recep.py" is for simulating the single neuron Hodgkin-Huxley data.

"Import_KineticModelBuiledr_Data.ipynb" is used for importing the Monte Carlo ismulation data and plotting against the deterministic model data.

"HH_KR_network.py" is for simulating the network of Hodgkin-Huxley equations using the KR model for coupling.

"HH_red_network.py" is for simulating the network of Hodgkin-Huxley equations using the reduced model for coupling.

"Opt_diff_ev_Kg_average.py" finds the optimised parameters related to the the slow and fast conductance variables in the reduced model.

"Opt_diff_ev_XU_multipleFreq.py" finds the optimised parameters related to the X and U variables in the reduced model.

"plotting_figures_for_paper.py" uses the data from the folder "Data" to make all simulation figures in the paper.