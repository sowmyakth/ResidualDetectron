"""Contains functions to perform detection, deblending and measurement
    on images.
"""
import sep
from btk import measure
import numpy as np
import scarlet
from scipy import spatial
import descwl
from astropy.table import vstack


def get_undetected(true_cat, meas_cent, obs_cond):
    """Returns bbox of galaxy that is undetected"""
    psf_sigma = obs_cond.psf_sigma_m
    pixel_scale = obs_cond.pixel_scale
    peaks = np.stack([true_cat['dx'], true_cat['dy']]).T
    z_tree = spatial.KDTree(peaks)
    tolerance = 10
    match = z_tree.query(meas_cent, distance_upper_bound=tolerance)
    undetected = np.setdiff1d(range(len(true_cat)), match)
    numer = true_cat['a_d']*true_cat['b_d']*true_cat['fluxnorm_disk'] + true_cat['a_b']*true_cat['b_b']*true_cat['fluxnorm_bulge']
    hlr = numer / (true_cat['fluxnorm_disk'] + true_cat['fluxnorm_bulge'])
    h = np.hypot(hlr, 1.18*psf_sigma)*2 / pixel_scale
    x0 = true_cat['dx'] - h/2
    y0 = true_cat['dy'] - h/2
    return x0[undetected], y0[undetected], h[undetected]


def get_random_shift(Args, number_of_objects, maxshift=None):
    """Returns a random shift from the center in x and y coordinates
    between 0 and max-shift (in arcseconds).
    """
    if not maxshift:
        maxshift = Args.stamp_size / 30.  # in arcseconds
    dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def resid_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = 1
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 25.3))
    if np.random.random() >= 0.9:
        q, = np.where(cond & (catalog['i_ab'] > 25.3) & (catalog['i_ab'] < 28))
    else:
        q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = vstack([catalog[np.random.choice(q_bright, size=1)],
                            catalog[np.random.choice(q,
                                                     size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_shift(Args, 1, maxshift=3*0.2)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    dr = np.random.uniform(3, 10)*0.2
    theta = np.random.uniform(0, 360) * np.pi / 180.
    dx2 = dr * np.cos(theta)
    dy2 = dr * np.sin(theta)
    blend_catalog['ra'][1] += dx2
    blend_catalog['dec'][1] += dy2
    return blend_catalog


def new_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = np.random.randint(1, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 25.3))
    q, = np.where(cond & (catalog['i_ab'] <= 26))
    blend_catalog = vstack([catalog[np.random.choice(q_bright, size=1)],
                            catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_shift(Args, number_of_objects + 1)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def resid_obs_conditions(Args, band):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band
    Args
        Args: Class containing parameters to generate blends
        band: filter name to get observing conditions for.
    Returns
        survey: WLD survey class with observing conditions.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name,
        filter_band=band)
    survey['zenith_psf_fwhm'] = 0.67
    survey['exposure_time'] = 5520
    return survey


class Scarlet_resid_params(measure.Measurement_params):
    iters = 400
    e_rel = .015
    detect_centers = True

    def make_measurement(self, data=None, index=None):
        return None

    def get_centers(self, image):
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        return np.stack((catalog['x'], catalog['y']), axis=1)

    def initialize(self, images, peaks,
                   bg_rms, iters, e_rel):
        """
        Deblend input images with scarlet
        Args:
            images: Numpy array of multi-band image to run scarlet on
                   [Number of bands, height, width].
            peaks: Array of x and y coordinates of centroids of objects in
                   the image [number of sources, 2].
            bg_rms: Background RMS value of the images [Number of bands]
            iters: Maximum number of iterations if scarlet doesn't converge
                   (Default: 200).
        e_rel: Relative error for convergence (Default: 0.015)
        Returns
            blend: scarlet.Blend object for the initialized sources
            rejected_sources: list of sources (if any) that scarlet was
            unable to initialize the image with.
        """
        sources = []
        for n, peak in enumerate(peaks):
            try:
                result = scarlet.ExtendedSource(
                    (peak[1], peak[0]),
                    images,
                    bg_rms)
                sources.append(result)
            except scarlet.source.SourceInitError:
                print("No flux in peak {0} at {1}".format(n, peak))
        blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
        return blend

    def multi_initialize(self, images, peaks,
                         bg_rms, iters, e_rel):
        """ Initializes scarlet MultiComponentSource at locations input as
        peaks in the (multi-band) input images.
        Args:
            images: Numpy array of multi-band image to run scarlet on
                    [Number of bands, height, width].
            peaks: Array of x and y coordinates of centroids of objects in
                   the image [number of sources, 2].
            bg_rms: Background RMS value of the images [Number of bands]
        Returns
            blend: scarlet.Blend object for the initialized sources
            rejected_sources: list of sources (if any) that scarlet was
                              unable to initialize the image with.
        """
        sources = []
        for n, peak in enumerate(peaks):
            try:
                result = scarlet.MultiComponentSource(
                    (peak[1], peak[0]),
                    images,
                    bg_rms)
                sources.append(result)
            except scarlet.source.SourceInitError:
                print("No flux in peak {0} at {1}".format(n, peak))
        blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
        return blend

    def scarlet_fit(self, images, peaks,
                    bg_rms, iters, e_rel):
        """Fits a scarlet model for the input image and centers"""
        try:
            blend = self.multi_initialize(images, peaks,
                                          bg_rms, iters, e_rel)
            blend.fit(iters, e_rel=e_rel)
        except (np.linalg.LinAlgError, ValueError):
            blend = self.initialize(images, peaks,
                                    bg_rms, iters, e_rel)
            try:
                blend.fit(iters, e_rel=e_rel)
            except(np.linalg.LinAlgError, ValueError):
                print("scarlet did not fit")
        return blend

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        blend_cat = data['blend_list'][index]
        if self.detect_centers:
            peaks = self.get_centers(images)
        else:
            peaks = np.stack((blend_cat['dx'], blend_cat['dy']), axis=1)
        bg_rms = [data['obs_condition'][i].mean_sky_level**0.5 for i in range(len(images))]
        blend = self.scarlet_fit(images, peaks,
                                 np.array(bg_rms), self.iters,
                                 self.e_rel)
        selected_peaks = [[src.center[1], src.center[0]]for src in blend.components]
        model = np.transpose(blend.get_model(), axes=(1, 2, 0))
        return [model, selected_peaks]
