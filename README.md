# DivBlurring
Diversity blurring to deblurr and denoise the medical images.

Recently, the progress of deep learning approaches has been enormous and has become
very useful in all research areas, such as biomedical image reconstruction. Improving the
spatial resolution of images acquired by standard microscopes is a rather challenging
task. Due to the physical barriers imposed by light diffraction phenomena, images acquired
in fluorescence microscopy setups are typically blurry and corrupted by electronic and
photon-counting noise. Removing noise and blur from observed noisy and blurry images is
the motivation in this report. By formulating the problem as an inverse image reconstruction
problem the task is thus to reconstruct the desired image from the noisy and blurry
data by using the generative approaches such as variational autoencoders (VAE)[1]. The
architecture of the variational autoencoder is helpful to estimate the distribution of the
unknown, noise- and blur-free data which we use to draw samples which are compared to
the given measured data. For this sake and for the particular application of covariancebased
fluorescence microscopy recently considered in [2], we propose a new approach which
is able to deblur and denoise the given input, up to a certain extent and we call it ”Div-
Blurring”. We detail the modelling considered and show some possible hybrid variations
where model-based priors are combined with fully data driven models to guarantee meaningful
solutions.

Prior to trian the model, create the environment with following code.
conda env create updDivNoising.yml

To run trianin the model, run the DivBlurring.ipynb file

