Code-to-Figure Mapping

A. Accuracy Comparison (MLP vs SNN)
* This is primarily from the results compilation of the training script and is not in the provided analysis code.

B-C. Neuron Activity Raster Plots (Dissipative vs Expansive)
* Corresponding File: `ANALYZE_B-C.py`
* Specific Function: `Net.forward()` where `spk_rec` data is recorded.
* Visualization: The raster plot itself likely requires additional plotting code.

D. Neurons Firing Rate
* Corresponding File: `ANALYZE_D-E-F.py`
* Specific Function: The part of `analyze_collective_activity()` that calculates the average firing rate.

E. Temporal Synchrony (Coefficient of Variation)
* Corresponding File: `ANALYZE_D-E-F.py`
* Specific Function: The part of `analyze_collective_activity()` that performs synchrony analysis.

F. Mean Spatial Correlation
* Corresponding File: `ANALYZE_D-E-F.py`
* Specific Function: `compute_neuron_correlations()`

G. Neural Silencing Robustness
* Corresponding File: `ANALYZE_G.py`
* Specific Function: `test_spike_perturbation_robustness()`

H. Layer-Label Information
* Corresponding File: `ANALYZE_layer_Vis_10.py`
* Specific Function: The calculation of MI(Layer, Label) within `calculate_information_plane_metrics()`.

I. Temporal Spikes Dimensionality
* Corresponding File: `ANALYZE_layer_Vis_10.py`
* Specific Function: The intrinsic dimensionality analysis, likely the result of `estimate_intrinsic_dim_PCA()`.