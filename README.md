# DivBlurring
Diversity blurring to deblurr and denoise the medical images.

Recently, the progress of deep learning approaches has been enormous and has become
very useful in all research areas, such as biomedical image reconstruction. Improving the
spatial resolution of images acquired by standard microscopes is a rather challenging
task. Removing noise and blur from observed noisy and blurry images is
the motivation. By formulating the problem as an inverse image reconstruction
problem the task is thus to reconstruct the desired image from the noisy and blurry
data by using the generative approaches such as variational autoencoders (VAE). The
architecture of the variational autoencoder is helpful to estimate the distribution of the
unknown, noise- and blur-free data which we use to draw samples which are compared to
the given measured data. We propose a new approach which is able to deblur and denoise the given input, up to a certain extent and we call it ”Div-
Blurring”. We detail the modelling considered and show some possible hybrid variations
where model-based priors are combined with fully data driven models to guarantee meaningful solutions.


Frist clone the repositary: `git clone https://github.com/saimuttavarapu/DivBlurring.git`.
Prior to trian the model, create the environment with following command: `conda env create DivBlurring.yml`.

Train the models by running `DivBlurring_training.ipynb`

Note: Verify the trained models are saved in respective directories.

For the prediction, run the `prediction.ipynb`

The architecture:
![teaserFigure]( https://github.com/saimuttavarapu/DivBlurring/blob/main/Supporting_imgs/DivBlurring_arc.png )

Resutls of DivBlurring with Positivity contrainint:
![teaserFigure](https://github.com/saimuttavarapu/DivBlurring/blob/main/Supporting_imgs/r2_DivBlurring_PCReg_1e3.png)

