"""Contains functions to perform detection, deblending and measurement
    on images.
"""
import sep
from btk import measure
import numpy as np


class Scarlet_params(measure.Measurement_params):
    iters = 200
    e_rel = .015

    def make_measurement(self, data=None, index=None):
        return None

    def get_deblended_images(self, data, index):
        """
        Deblend input images with scarlet
        Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y cordinate of cntroids of objects in the image.
               [number of sources, 2]
        bg_rms: Background RMS value of the images [Number of bands]
        iters: Maximum number of iterations if scarlet doesn't converge
               (Default: 200).
        e_rel: Relative error for convergence (Deafult: 0.015)
        Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
        unable to initlaize the image with.
        """
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        blend_cat = data['blend_list'][index]
        peaks = np.stack((blend_cat['dx'], blend_cat['dy']), axis=1)
        bg_rms = data['sky_level'][index]**0.5
        blend, rejected_sources = scarlet_initialize(images, peaks,
                                                     bg_rms, self.iters,
                                                     self.e_rel)
        im = []
        for m in range(len(blend.sources)):
            oth_indx = np.delete(range(len(blend.sources)), m)
            model_oth = np.zeros_like(images)
            for i in oth_indx:
                model_oth += blend.get_model(k=i)
            im.append(np.transpose(images - model_oth, axes=(1, 2, 0)))
        return np.array(im)


def scarlet_initialize(images, peaks,
                       bg_rms, iters, e_rel):
    """ Intializes scarlet ExtendedSource at locations specified as peaks
    in the (multi-band) input images.
    Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y cordinate of cntroids of objects in the image.
               [number of sources, 2]
        bg_rms: Background RMS value of the images [Number of bands]
    Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
                          unable to initlaize the image with.
    """
    import scarlet
    sources, rejected_sources = [], []
    for n, peak in enumerate(peaks):
        try:
            result = scarlet.ExtendedSource(
                (peak[1], peak[0]),
                images,
                bg_rms)
            sources.append(result)
        except scarlet.source.SourceInitError:
            rejected_sources.append(n)
            print("No flux in peak {0} at {1}".format(n, peak))
    blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
    blend.fit(iters, e_rel=e_rel)
    return blend, rejected_sources
