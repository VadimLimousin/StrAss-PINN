# StrAss-PINN
Attachment to the paper "Deep learning in the abyss:  a stratified Physics Informed Neural Network for data assimilation"

## Abstract
The reconstruction of deep ocean currents is a major challenge in data assimilation due to13the  scarcity  of  interior  data.   In  this  work,  we  present  a  proof  of  concept  for  deep  ocean flow reconstruction using a Physics-Informed Neural Network (PINN), a machine learning approach that offers an alternative to traditional data assimilation methods.  We introduce an efficient algorithm called StrAssPINN (for Stratified PINNs), which assigns a separate network to each layer of the ocean model while allowing them to interact during training. The neural network takes spatiotemporal coordinates as input and predicts the velocity field at those points.  Using a SIREN architecture (a multilayer perceptron with sine activation functions), which has proven effective in various contexts, the network is trained using both available observational data and dynamical priors enforced at several colocation points.  We apply this method to pseudo-observed ocean data generated from a 3-layer quasi-geostrophic model, where the pseudo-observations include surface-level data akin to SWOT observations of sea surface height, interior data similar to ARGO floats, and a limited number of deep ARGO-like measurements in the lower layers.  Our approach successfully reconstructs ocean flows in both the interior and surface layers, demonstrating a strong ability to resolve key ocean  mesoscale  features,  including  vortex  rings,  eastward  jets  associated  with  potential vorticity  fronts,  and  smoother  Rossby  waves.   This  work  serves  as  a  prelude  to  applying StrAss-PINNto real-world observational data.

## Dataset

Data files required to run these codes are available on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15074965.svg)](https://doi.org/10.5281/zenodo.15074965). The data requirements are: 
  • To run the introductory notebook: params.in, vars.nc, coords_128_100j, mask_128_100j, psi1_128_100j, std_128_100j
  • To run the full method: params.in, vars.nc, coords, mask, psi1, std
