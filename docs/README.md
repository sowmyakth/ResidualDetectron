## ResidualDetectron

Due to the increased depth at which the Large Synoptic Survey Telescope (LSST) will be observing, a large fraction of the observed galaxies will overlap with images of other objects. The extent of blending will be comparable to the overlapping object in the HSC Ultra-Deep Field shown below.
<img src="imgs/hsc_image.png" alt="HSC COSMOS Ultra-Deep Field" width="200"/>

A significant fraction of these overlapping galaxies will not be recognized as blended object by the LSST Science Pipeline. This could result in several issues:
* Detection errors (objects not detected or incorrect initial centroid location).
* Errors in separation of flux at pixel level (“deblending”).
* Measurement of shape, photometry, … errors.
* Selection effects.

We explore in this project an iterative method to detect objects using Neural Networks. We run a region based Convolutional Neural Network on the model residuals of the current LSST "deblending" algorithm. Let's discuss this in detail.

Could unrecognized blends be detected on the residual images?
Aim: Run Mask R-CNN detection network, with residual images + the Scarlet model as input, to predict undetected source locations.


