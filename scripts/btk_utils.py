"""Contains functions to perform detection, deblending and measurement
    on images with BlendingToolKit(btk).
"""
import sep
import btk
import btk.config
import os
import numpy as np
import scarlet
from scipy import spatial
import descwl
import matplotlib.pyplot as plt
import astropy.table
from mrcnn import utils
import mrcnn.config_btk_only
from functools import partial
import multiprocessing


def get_ax(rows=1, cols=1, size=4):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def resid_merge_centers(det_cent, bbox,
                        distance_upper_bound=1, center_shift=0):
    """Combines centers from detection algorithm and iteratively
    detected centers. Also corrects for shift of 4 pixels in center
    Args:
        det_cent: centers detected by detection algorithm.
        bbox: Edges of ResidDetectron bounding box (y1, x1, y2, x2).
        distance_upper_bound: If network prediction is within this distance of
                              a det_cent, select the network prediction and
                              remove det_cent from final merged predictions.
        center_shift: Value to offset the bbox centers by. Applicable if
            padding was applied to the residual image causing detected centers
            and bounding box centers to offset.
    """
    # remove duplicates
    if len(bbox) == 0:
        if len(det_cent) == 0:
            return []
        else:
            return np.unique(det_cent, axis=0)
    q, = np.where(
        (bbox[:, 0] > 3+center_shift) & (bbox[:, 1] > 3+center_shift))
    # centers of bbox as mean of edges
    resid_det = np.dstack([np.mean(bbox[q, 1::2], axis=1) - center_shift,
                           np.mean(bbox[q, ::2], axis=1) - center_shift])[0]
    if len(det_cent) == 0:
        return resid_det
    unique_det_cent = np.unique(det_cent, axis=0)
    z_tree = spatial.KDTree(unique_det_cent)
    resid_det = resid_det.reshape(-1, 2)
    match = z_tree.query(resid_det,
                         distance_upper_bound=distance_upper_bound)
    trim = np.setdiff1d(range(len(unique_det_cent)), match[1])
    trim_det_cent = [unique_det_cent[i] for i in trim]
    if len(trim_det_cent) == 0:
        return resid_det
    iter_det = np.vstack([trim_det_cent, resid_det])
    return iter_det


def get_undetected(true_cat_all, meas_cent, obs_cond,
                   i_mag_lim=30, distance_upper_bound=10):
    """Returns bounding boxes for galaxies that could have been detected but
    were undetected.
    All galaxies brighter than detctect_mag in i band are labelled as
    detectable. If a detectable galaxy is not matched to meas_cent then it is
    set as undetected galaxy. The bounding boxes are square with height set as
    twice the PSF convolved HLR. Since CatSim catalog has separate bulge and
    disk parameters, the galaxy HLR is approximated as the flux weighted
    average of bulge and disk HLR.

    The function returns the x and y coordinates of the lower left corner of
    box, and the height of each undetected galaxy. A galaxy is marked as
    undetected if no detected center lies within distance_upper_bound of it's
    true center.
    Args:
        true_cat: CatSim-like catalog of true galaxies.
        meas_cent: ndarray of x and y coordinates of detected centers.
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        distance_upper_bound: Distance up-to which a detected center can be
                              matched to a true center.
        i_mag_lim(float): galaxies with magnitude brighter than is are said to
            be detectable.

    Returns:
        Bounding box of undetected galaxies.

    """
    true_cat = true_cat_all[
        (true_cat_all['not_drawn_i'] != 1) &
        (true_cat_all['i_ab'] <= i_mag_lim)]
    psf_sigma = obs_cond.zenith_psf_fwhm*obs_cond.airmass**0.6
    pixel_scale = obs_cond.pixel_scale
    peaks = np.stack(
        [np.array(true_cat['dx']), np.array(true_cat['dy'])]).T
    if len(peaks) == 0:
        undetected = range(len(true_cat))
    else:
        z_tree = spatial.KDTree(peaks)
        meas_cent = np.array(meas_cent).reshape(-1, 2)
        match = z_tree.query(meas_cent,
                             distance_upper_bound=distance_upper_bound)
        undetected = np.setdiff1d(range(len(true_cat)), match[1])
    numer = true_cat['a_d']*true_cat['b_d']*true_cat['fluxnorm_disk'] + true_cat['a_b']*true_cat['b_b']*true_cat['fluxnorm_bulge']
    hlr = numer / (true_cat['fluxnorm_disk'] + true_cat['fluxnorm_bulge'])
    assert ~np.any(np.isnan(hlr)), "FOUND NAN"
    h = np.hypot(hlr, 1.18*psf_sigma)*2 / pixel_scale
    x0 = true_cat['dx'] - h/2
    y0 = true_cat['dy'] - h/2
    h_arr = np.array(h, dtype=np.int32)[undetected]
    x0_arr = np.array(x0, dtype=np.int32)[undetected]
    y0_arr = np.array(y0, dtype=np.int32)[undetected]
    return x0_arr, y0_arr, h_arr


def get_random_shift(Args, number_of_objects, maxshift=None):
    """Returns a random shift from the center in x and y coordinates
    between 0 and max-shift (in arcseconds).
    """
    if not maxshift:
        maxshift = Args.stamp_size / 10.  # in arcseconds
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
    blend_catalog = astropy.table.vstack(
        [catalog[np.random.choice(q_bright, size=1)],
         catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # Add small shift so that center does not perfectly align with stamp center
    dx, dy = get_random_shift(Args, 1, maxshift=3*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    dr = np.random.uniform(3, 10)*Args.pixel_scale
    theta = np.random.uniform(0, 360) * np.pi / 180.
    dx2 = dr * np.cos(theta)
    dy2 = dr * np.sin(theta)
    blend_catalog['ra'][1] += dx2
    blend_catalog['dec'][1] += dy2
    return blend_catalog


def resid_general_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    At least one bright galaxy (i<=24) is always selected.
    """
    number_of_objects = np.random.randint(0, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 24))
    if np.random.random() >= 0.9:
        q, = np.where(cond & (catalog['i_ab'] < 28))
    else:
        q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = astropy.table.vstack(
        [catalog[np.random.choice(q_bright, size=1)],
         catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # keep number density of objects constant
    #maxshift = Args.stamp_size/30.*number_of_objects**0.5
    maxshift = Args.stamp_size/20.*number_of_objects**0.5
    dx, dy = get_random_shift(Args, number_of_objects + 1,
                              maxshift=maxshift)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # Shift center of all objects so that the blend isn't exactly in the center
    dx, dy = get_random_shift(Args, 1, maxshift=5*Args.pixel_scale)
    return blend_catalog


def resid_general_sampling_function_large(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    At least one bright galaxy (i<=24) is always selected.
    Preferrentially select larger galaxies. Also offset from center is larger.
    """
    number_of_objects = np.random.randint(1, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 3) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 24))
    q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = astropy.table.vstack(
        [catalog[np.random.choice(q_bright, size=1)],
         catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # keep number density of objects constant
    maxshift = Args.stamp_size/20.*number_of_objects**0.5
    dx, dy = get_random_shift(Args, number_of_objects + 1,
                              maxshift=maxshift)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # Shift center of all objects so that the blend isn't exactly in the center
    dx, dy = get_random_shift(Args, 1, maxshift=30*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
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
    blend_catalog = astropy.table.vstack(
        [catalog[np.random.choice(q_bright, size=1)],
         catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_shift(Args, number_of_objects + 1)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def get_wld_catalog(path):
    """Returns pre-run wld catalog for group identification"""
    wld_catalog_name = os.path.join(
        path, 'test_group_catalog.fits')
    wld_catalog = astropy.table.Table.read(wld_catalog_name, format='fits')
    # selected_gal = wld_catalog[
    #    (wld_catalog['sigma_m'] < 2) & (wld_catalog['ab_mag'] < 25.3)]
    return wld_catalog


def group_sampling_function(Args, catalog, min_group_size=2):
    """Blends are defined from *groups* of galaxies from the Cat-Sim like
    catalog previously analyzed with WLD. Function selects galaxies
    Note: the pre-run WLD images are not used here. We only use the pre-run
    catalog (in i band) to identify galaxies that belong to a group.

    Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    if not hasattr(Args, 'wld_catalog'):
        raise Exception(
            "A pre-run WLD catalog should be input as Args.wld_catalog")
    else:
        wld_catalog = Args.wld_catalog
    group_ids = np.unique(
        wld_catalog['grp_id'][wld_catalog['grp_size'] >= min_group_size])
    group_id = np.random.choice(group_ids)
    ids = wld_catalog['db_id'][wld_catalog['grp_id'] == group_id]
    blend_catalog = astropy.table.vstack(
        [catalog[catalog['galtileid'] == i] for i in ids])
    blend_catalog['ra'] -= np.mean(blend_catalog['ra'])
    blend_catalog['dec'] -= np.mean(blend_catalog['dec'])
    # convert ra dec from degrees to arcsec
    blend_catalog['ra'] *= 3600
    blend_catalog['dec'] *= 3600
    # Add small shift so that center does not perfectly align with stamp center
    dx, dy = get_random_shift(Args, 1, maxshift=3*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # make sure galaxy centers don't lie too close to edge
    cond1 = np.abs(blend_catalog['ra']) < Args.stamp_size/2. - 5
    cond2 = np.abs(blend_catalog['dec']) < Args.stamp_size/2. - 5
    no_boundary = blend_catalog[cond1 & cond2]
    if len(no_boundary) == 0:
        return no_boundary
    # make sure number of galaxies in blend is less than Args.max_number
    num = min([len(no_boundary), Args.max_number])
    select = np.random.choice(range(len(no_boundary)), num, replace=False)
    return no_boundary[select]


def group_sampling_function_numbered(Args, catalog):
    """Blends are defined from *groups* of galaxies from a CatSim-like
    catalog previously analyzed with WLD.

    This function requires a parameter, group_id_count, to be input in Args
    along with the wld_catalog which tracks the group id returned. Each time
    the generator is called,1 gets added to the count. If the count is
    larger than the number of groups input, the generator is forced to exit.

    The group is centered on the middle of the postage stamp.
    This function only draws galaxies whose centers lie within 1 arcsec the
    postage stamp edge, which may cause the number of galaxies in the blend to
    be smaller than the group size.The pre-run wld catalog must be defined as
    Args.wld_catalog.

    Note: the pre-run WLD images are not used here. We only use the pre-run
    catalog (in i band) to identify galaxies that belong to a group.

    Args:
        Args: Class containing input parameters.
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        Catalog with entries corresponding to one blend.
    """
    if not hasattr(Args, 'wld_catalog'):
        raise Exception(
            "A pre-run WLD catalog should be input as Args.wld_catalog")
    if not hasattr(Args, 'group_id_count'):
        raise NameError("An integer specifying index of group_id to draw must"
                        "be input as Args.group_id_count")
    elif not isinstance(Args.group_id_count, int):
        raise ValueError("group_id_count must be an integer")
    # randomly sample a group.
    group_ids = np.unique(
        Args.wld_catalog['grp_id'][
            (Args.wld_catalog['grp_size'] >= 2) &
            (Args.wld_catalog['grp_size'] <= Args.max_number)])
    if Args.group_id_count >= len(group_ids):
        message = "group_id_count is larger than number of groups input"
        raise GeneratorExit(message)
    else:
        group_id = group_ids[Args.group_id_count]
        Args.group_id_count += 1
    # get all galaxies belonging to the group.
    # make sure some group or galaxy was not repeated in wld_catalog
    ids = np.unique(
        Args.wld_catalog['db_id'][Args.wld_catalog['grp_id'] == group_id])
    blend_catalog = astropy.table.vstack(
        [catalog[catalog['galtileid'] == i] for i in ids])[:Args.max_number]
    # Set mean x and y coordinates of the group galaxies to the center of the
    # postage stamp.
    blend_catalog['ra'] -= np.mean(blend_catalog['ra'])
    blend_catalog['dec'] -= np.mean(blend_catalog['dec'])
    # convert ra dec from degrees to arcsec
    blend_catalog['ra'] *= 3600
    blend_catalog['dec'] *= 3600
    # Add small random shift so that center does not perfectly align with stamp
    # center
    dx, dy = btk.create_blend_generator.get_random_center_shift(
        Args, 1, maxshift=5*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # make sure galaxy centers don't lie too close to edge
    cond1 = np.abs(blend_catalog['ra']) < Args.stamp_size/2. - 1
    cond2 = np.abs(blend_catalog['dec']) < Args.stamp_size/2. - 1
    no_boundary = blend_catalog[cond1 & cond2]
    message = ("Number of galaxies greater than max number of objects per"
               f"blend. Found {len(no_boundary)}, expected <= {Args.max_number}")
    assert len(no_boundary) <= Args.max_number, message
    return no_boundary


def custom_obs_condition(Args, band):
    """Returns observing conditions from the WLD package
    for a given survey_name and band with a small offset from
    the default parameters.
    Args
        Args: Class containing parameters to generate blends
        band: filter name to get observing conditions for.
    Returns
        survey: WLD survey class with observing conditions.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name,
        filter_band=band)
    survey['exposure_time'] += np.random.uniform(-50, 50)
    survey['zenith_psf_fwhm'] += np.random.uniform(-0.05, 0.05)
    return survey


def custom_selection_function(catalog):
    """Apply selection cuts to the input catalog"""
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    q, = np.where((a <= 4) & (a > 0.2) & (catalog['i_ab'] <= 27))
    return catalog[q]


def basic_selection_function(catalog):
    """Apply selection cuts to the input catalog"""
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    q, = np.where((a <= 2) & (a > 0.2) & (catalog['i_ab'] <= 26.5))
    return catalog[q]


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
    # survey['mirror_diameter'] = 0
    return survey


def scarlet_initialize(images, peaks,
                       bg_rms):
    """ Deblend input images with scarlet
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


def scarlet1_initialize(images, peaks, psfs, variances, bands):
    """ Deblend input images with scarlet
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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=0.8),
                            shape=(None, 41, 41))
    model_frame = scarlet.Frame(images.shape, psfs=model_psf, channels=bands)
    observation = scarlet.Observation(images, psfs=scarlet.PSF(psfs),
                                      weights=1./variances,
                                      channels=bands).match(model_frame)
    sources = []
    for n, peak in enumerate(peaks):
        result = scarlet.ExtendedSource(model_frame, (peak[1], peak[0]),
                                        observation, symmetric=True,
                                        monotonic=True, thresh=1,
                                        shifting=True)
        sed = result.sed
        morph = result.morph
        if np.all([s < 0 for s in sed]) or np.sum(morph) == 0:
            raise ValueError("Incorrectly initialized")
        sources.append(result)
    blend = scarlet.Blend(sources, observation)
    return blend, observation


def scarlet_multi_initialize(images, peaks,
                             bg_rms):
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


def scarlet1_multi_initialize(images, peaks, psfs, variances, bands):
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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=0.8),
                            shape=(None, 41, 41))
    model_frame = scarlet.Frame(images.shape, psfs=model_psf, channels=bands)
    observation = scarlet.Observation(images, psfs=scarlet.PSF(psfs),
                                      weights=1./variances,
                                      channels=bands).match(model_frame)
    sources = []
    for n, peak in enumerate(peaks):
        result = scarlet.MultiComponentSource(model_frame, (peak[1], peak[0]),
                                              observation, symmetric=True,
                                              monotonic=True, thresh=1,
                                              shifting=True)
        for i in range(result.n_sources):
            sed = result.components[i].sed
            morph = result.components[i].morph
            if np.all([s < 0 for s in sed]) or np.sum(morph) == 0:
                raise ValueError("Incorrectly initialized")
        sources.append(result)
    blend = scarlet.Blend(sources, observation)
    return blend, observation


def scarlet_fit(images, peaks, psfs, variances,
                bands, iters, e_rel, f_rel):
    """Fits a scarlet model for the input image and centers"""
    scarlet_multi_fit = 1
    try:
        blend, observation = scarlet1_multi_initialize(
            images, peaks, psfs, variances, np.array(bands, dtype=str))
        blend.fit(iters, e_rel=e_rel, f_rel=f_rel)
    except (np.linalg.LinAlgError, ValueError) as e:
        print("multi component initialization failed \n", e)
        blend, observation = scarlet1_initialize(
            images, peaks, psfs, variances, np.array(bands, dtype=str))
        try:
            blend.fit(iters, e_rel=e_rel, f_rel=f_rel)
            scarlet_multi_fit = 0
        except(np.linalg.LinAlgError, ValueError) as e:
            print("scarlet did not fit \n", e)
            scarlet_multi_fit = -1
    return blend, observation, scarlet_multi_fit


class Scarlet_resid_params(btk.measure.Measurement_params):
    def __init__(self, iters=200, e_rel=.015, f_rel=1e-6,
                 detect_centers=True, detect_coadd=False,
                 *args, **kwargs):
        super(Scarlet_resid_params, self).__init__(*args, **kwargs)
        self.iters = iters
        self.e_rel = e_rel
        self.detect_centers = detect_centers
        self.detect_coadd = detect_coadd
        self.f_rel = f_rel

    def get_centers_coadd(self, image):
        """Runs SEP on coadd of input image and returns detected centroids
        Args:
            image: Input image (multi-band) to perform detection on
                   [bands, x, y].
        Returns:
            x and y coordinates of detected centroids.
        """
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect, bw=32, bh=32)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        q, = np.where((catalog['x'] > 0) & (catalog['y'] > 0))
        return np.stack((catalog['x'][q], catalog['y'][q]), axis=1)

    def get_centers_i_band(self, image):
        """Runs SEP on i band of input image and returns detected centroids
        Args:
            image: Input image (multi-band) to perform detection on
                   [bands, x, y].
        Returns:
            x and y coordinates of detected centroids.
        """
        # simple average for detection
        detect = np.array(image[3, :, :], dtype=np.float32)
        bkg = sep.Background(detect, bw=32, bh=32)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        q, = np.where((catalog['x'] > 0) & (catalog['y'] > 0))
        return np.stack((catalog['x'][q], catalog['y'][q]), axis=1)

    def get_deblended_images(self, data, index, peaks=None):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        images[np.isnan(images)] = 0
        bands = []
        psf_stamp_size = 41
        psfs = np.zeros((len(images), psf_stamp_size, psf_stamp_size),
                        dtype=np.float32)
        variances = np.zeros_like(images)
        for i in range(len(images)):
            bands.append(data['obs_condition'][index][i].filter_band)
            psf, mean_sky_level = get_psf_sky(
                data['obs_condition'][index][i], psf_stamp_size)
            psfs[i] = psf
            variances[i] = images[i] + mean_sky_level
        if self.detect_centers:
            if self.detect_coadd:
                peaks = self.get_centers_coadd(images)
            else:
                peaks = self.get_centers_i_band(images)
        if len(peaks) == 0:
            return {'scarlet_model': data['blend_images'][index],
                    'scarlet_peaks': peaks, 'scarlet_multi_fit': 0}
        blend, observation, sf = scarlet_fit(
            images, peaks, psfs, variances, bands,
            self.iters, self.e_rel, self.f_rel)
        #selected_peaks = [[src.center[1], src.center[0]]for src in blend.components]
        temp_model = np.zeros_like(data['blend_images'][index])
        try:
            #model = np.transpose(blend.get_model(), axes=(1, 2, 0))
            selected_peaks = []
            for k, component in enumerate(blend):
                y, x = component.center
                selected_peaks.append([x, y])
                model = component.get_model()
                model_ = observation.render(model)
                temp_model += np.transpose(model_, axes=(1, 2, 0))
        except(ValueError):
            print("Unable to create scarlet model")
            return {'scarlet_model': temp_model, 'scarlet_peaks': [],
                    'scarlet_multi_fit': sf}
        return {'scarlet_model': temp_model, 'scarlet_peaks': selected_peaks,
                'scarlet_multi_fit': sf}


def get_psf_sky(obs_cond, psf_stamp_size):
    """Returns PSF image and mean background sky level for input obs_condition.
    Args:
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        psf_stamp_size: Size of image to draw PSF model into (pixels).
    Returns:
        PSF image and mean background sky level
    """
    mean_sky_level = obs_cond.mean_sky_level
    psf = obs_cond.psf_model
    psf_image = psf.drawImage(
        nx=psf_stamp_size,
        ny=psf_stamp_size).array
    return psf_image, mean_sky_level


def get_stack_input(image, obs_cond, psf_stamp_size, detect_coadd):
    """Returns input for running stack detection on either coadd image or
    i band.
    Args:
        image: Input image (multi-band) to perform detection on
               [bands, x, y].
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        psf_stamp_size: Size of image to draw PSF model into (pixels).
        detect_coadd: If True then detection (and measurement) is
                      performed on coadd over bands
    Returns
        image, variance image and psf image
    """
    if detect_coadd:
        input_image = np.zeros(image.shape[0:2], dtype=np.float32)
        variance_image = np.zeros(image.shape[0:2], dtype=np.float32)
        for i in range(len(obs_cond)):
            psf_image, mean_sky_level = get_psf_sky(obs_cond[i],
                                                    psf_stamp_size)
            variance_image += image[:, :, i] + mean_sky_level
            input_image += image[:, :, i]
    else:
        i = 3  # detection in i band
        psf_image, mean_sky_level = get_psf_sky(obs_cond[i],
                                                psf_stamp_size)
        variance_image = image[:, :, i] + mean_sky_level
        variance_image = np.array(variance_image, dtype=np.float32)
        input_image = np.array(image[:, :, i], dtype=np.float32)
    # since PSF is same for all bands, PSF of coadd is the same
    return input_image, variance_image, psf_image


def get_stack_catalog(image, obs_cond, detect_coadd=False,
                      psf_stamp_size=41, min_pix=1,
                      thr_value=5, bkg_bin_size=32):
    """Perform detection, deblending and measurement on the i band image of
    the blend image for input index in the batch.
    Args:
        image: Input image (multi-band) to perform detection on
               [bands, x, y].
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        detect_coadd: If True then detection (and measurement) is
                      performed on coadd over bands
    Returns
        Stack detection (+ measurement) output catalog
    """
    image_array, variance_array, psf_image = get_stack_input(
        image, obs_cond, psf_stamp_size, detect_coadd)
    psf_array = psf_image.astype(np.float64)
    cat = btk.utils.run_stack(
        image_array, variance_array, psf_array, min_pix=min_pix,
        bkg_bin_size=bkg_bin_size, thr_value=thr_value)
    cat_chldrn = cat[
        (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
    cat_chldrn = cat_chldrn.copy(deep=True)
    return cat_chldrn.asAstropy()


def get_stack_centers(catalog):
    """Returns stack detected centroids from detection catalog.
    Args:
        catalog: Stack detection output catalog
    Returns:
        x and y coordinates of detected centroids.
    """
    xs = catalog['base_SdssCentroid_y']
    ys = catalog['base_SdssCentroid_x']
    q, = np.where((xs > 0) & (ys > 0))
    return np.stack((ys[q], xs[q]), axis=1)


class Stack_iter_params(btk.measure.Measurement_params):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5
    psf_stamp_size = 41
    iters = 200
    e_rel = .015

    def __init__(self, detect_coadd=True, *args, **kwargs):
        super(Stack_iter_params, self).__init__(*args, **kwargs)
        self.detect_coadd = detect_coadd
        self.catalog = {}

    def get_only_peaks(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        catalog = get_stack_catalog(data['blend_images'][index],
                                    data['obs_condition'][index],
                                    detect_coadd=self.detect_coadd,
                                    psf_stamp_size=self.psf_stamp_size,
                                    min_pix=self.min_pix,
                                    bkg_bin_size=self.bkg_bin_size,
                                    thr_value=self.thr_value)
        peaks = get_stack_centers(catalog)
        return peaks

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        catalog = get_stack_catalog(data['blend_images'][index],
                                    data['obs_condition'][index],
                                    detect_coadd=self.detect_coadd,
                                    psf_stamp_size=self.psf_stamp_size,
                                    min_pix=self.min_pix,
                                    bkg_bin_size=self.bkg_bin_size,
                                    thr_value=self.thr_value)

        self.catalog[index] = catalog
        peaks = get_stack_centers(catalog)
        if len(peaks) == 0:
            print("Unable to create scarlet model, no peaks")
            temp_model = np.zeros_like(data['blend_images'][index])
            return {'scarlet_model': temp_model, 'scarlet_peaks': []}
        bg_rms = [data['obs_condition'][index][i].mean_sky_level**0.5 for i in range(len(images))]
        try:
            blend = scarlet_fit(images, peaks,
                                np.array(bg_rms), self.iters,
                                self.e_rel)
            selected_peaks = [[src.center[1], src.center[0]]for src in blend.components]
            model = np.transpose(blend.get_model(), axes=(1, 2, 0))
        except(ValueError, IndexError) as e:
            print("Unable to create scarlet model")
            print(e)
            temp_model = np.zeros_like(data['blend_images'][index])
            return {'scarlet_model': temp_model, 'scarlet_peaks': []}
        return {'scarlet_model': model, 'scarlet_peaks': selected_peaks}

    def make_measurement(self, data, index):
        """ Returns catalog from the deblending step which involved performing
        detection, deblending and measurement on the i band image of
        the blend image for input index in the batch using the DM stack.
         """
        return self.catalog[index]


def make_meas_generator(catalog_name, batch_size, max_number,
                        sampling_function, selection_function=None,
                        wld_catalog=None, meas_params=None,
                        obs_condition=None, multiprocess=False):
        """
        Creates the default btk.meas_generator for input catalog
        Args:
            catalog_name: CatSim like catalog to draw galaxies from.
            max_number: Maximum number of galaxies per blend.
            sampling_function: Function describing how galaxies are drawn from
                               catalog_name.
            wld_catalog: A WLD pre-run astropy table. Used if sampling function
                         requires grouped objects.
        """
        # Load parameters
        param = btk.config.Simulation_params(
            catalog_name, max_number=max_number, stamp_size=25.6,
            batch_size=batch_size, draw_isolated=False, seed=199)
        if wld_catalog:
            print("wld catalog provided:")
            param.wld_catalog = wld_catalog
            param.group_id_count = 0
        print("setting seed", param.seed)
        np.random.seed(param.seed)
        # Load input catalog
        catalog = btk.get_input_catalog.load_catalog(
            param, selection_function=selection_function)
        # Generate catalogs of blended objects
        blend_generator = btk.create_blend_generator.generate(
            param, catalog, sampling_function)
        # Generates observing conditions
        if obs_condition is None:
            obs_condition = resid_obs_conditions
        observing_generator = btk.create_observing_generator.generate(
            param, obs_condition)
        # Generate images of blends in all the observing bands
        draw_blend_generator = btk.draw_blends.generate(
            param, blend_generator, observing_generator)
        if meas_params is None:
            print("Setting scarlet_resid_paramsa as meas_params")
            meas_params = Scarlet_resid_params()
        if multiprocess:
            cpus = multiprocessing.cpu_count()
        else:
            cpus = 1
        meas_generator = btk.measure.generate(
            meas_params, draw_blend_generator, param,
            multiprocessing=multiprocess, cpus=cpus)
        return meas_generator


class ResidDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self, meas_generator, norm_val=None,
                 augmentation=False, i_mag_lim=30, input_pull=False,
                 norm_limit=None, input_model_mapping=False, stretch=2731,
                 *args, **kwargs):
        super(ResidDataset, self).__init__(*args, **kwargs)
        self.meas_generator = meas_generator
        self.augmentation = augmentation
        self.i_mag_lim = i_mag_lim
        self.input_pull = input_pull
        self.input_model_mapping = input_model_mapping
        self.stretch = stretch
        if norm_val:
            self.mean1 = norm_val[0]
            self.std1 = norm_val[1]
            self.mean2 = norm_val[2]
            self.std2 = norm_val[3]
        else:
            self.mean1 = 1.6361405416087091
            self.std1 = 416.16687641284665
            self.mean2 = 63.16814480535191
            self.std2 = 2346.133101333463
        self.norm_limit = norm_limit
        print("Input normalized with", norm_val)

    def load_data(self, count=None):
        """loads training and test input and output data
        Keyword Arguments:
            filename -- Numpy file where data is saved
        """
        if not count:
            count = 240
        self.load_objects(count)
        print("Loaded {} blends".format(count))

    def load_objects(self, count):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("resid", 1, "object")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            self.add_image("resid", image_id=i, path=None,
                           object="object")

    def normalize_images(self, images):
        images[:, :, :, 0:6] = (images[:, :, :, 0:6] - self.mean1)/self.std1
        images[:, :, :, 6:12] = (images[:, :, :, 6:12] - self.mean2)/self.std2
        return images

    def normalize_images_with_lim(self, images):
        """Saturate at +-lim"""
        images[:, :, :, 0:6] = (images[:, :, :, 0:6] - self.mean1)/self.std1
        images[:, :, :, 6:12] = (images[:, :, :, 6:12] - self.mean2)/self.std2
        if isinstance(self.norm_limit, list):
            assert len(self.norm_limit) == 2, "norm_limit must be None or "\
                "2-element list"
            images[images < self.norm_limit[0]] = self.norm_limit[0]
            images[images > self.norm_limit[1]] = self.norm_limit[1]
        else:
            assert self.norm_limit is None, "norm_limit must be None or "\
                "2-element list"
        return images

    def augment_bbox(self, bboxes, end_pixel):
        mult_y = np.array([0, 0, 1, 1])
        mult_x = np.array([0, 1, 0, 1])
        h0 = (bboxes[:, 2] - bboxes[:, 0]) / 2.
        x0 = np.mean(bboxes[:, 1::2], axis=1)
        y0 = np.mean(bboxes[:, ::2], axis=1)
        aug_bbox = np.zeros((4, len(bboxes), 4), dtype=np.int32)
        for i in range(len(mult_x)):
            new_x0 = np.abs(end_pixel*mult_x[i] - x0)
            new_y0 = np.abs(end_pixel*mult_y[i] - y0)
            new_x0[x0 == 0.5] = 0.5
            new_y0[x0 == 0.5] = 0.5
            new_bbox = np.array(
                [new_y0 - h0, new_x0 - h0, new_y0 + h0, new_x0 + h0])
            new_bbox = np.transpose(new_bbox, axes=(1, 0,))
            aug_bbox[i] = new_bbox
        assert ~np.any(np.isnan(aug_bbox)), "FOUND NAN"
        return aug_bbox

    def augment_data(self, images, bboxes, class_ids):
        """Performs data augmentation by performing rotation and reflection"""
        aug_image = np.stack([images[:, :, :],
                              images[:, ::-1, :],
                              images[::-1, :, :],
                              images[::-1, ::-1, :]])
        aug_bbox = self.augment_bbox(bboxes, images.shape[1] - 1)
        aug_class = np.stack([class_ids, class_ids, class_ids, class_ids])
        return aug_image, aug_bbox, aug_class

    def load_input(self):
        """Generates image + bbox for undetected objects if any"""
        output, deb, _ = next(self.meas_generator)
        self.batch_blend_list = output['blend_list']
        self.scarlet_multi_fit = []
        self.obs_cond = output['obs_condition']
        input_images, input_bboxes, input_class_ids = [], [], []
        self.det_cent, self.true_cent = [], []
        for i in range(len(output['blend_list'])):
            blend_images = output['blend_images'][i]
            blend_list = output['blend_list'][i]
            model_images = deb[i]['scarlet_model']
            model_images[np.isnan(model_images)] = 0
            detected_centers = deb[i]['scarlet_peaks']
            self.det_cent.append(detected_centers)
            self.scarlet_multi_fit.append(deb[i]['scarlet_multi_fit'])
            cent_t = [np.array(blend_list['dx']), np.array(blend_list['dy'])]
            self.true_cent.append(np.stack(cent_t).T)
            resid_images = blend_images - model_images
            if self.input_pull:
                bg_rms = [c.mean_sky_level for c in self.obs_cond[i]]
                resid_images /= np.sqrt(np.array(bg_rms) + blend_images)
            if self.input_model_mapping:
                # stretch = 2000  # 0.1
                Q = 0.5  # 3
                model_images = np.arcsinh(Q*model_images/self.stretch)/Q
            image = np.dstack([resid_images, model_images])
            x, y, h = get_undetected(blend_list, detected_centers,
                                     self.obs_cond[i][3], self.i_mag_lim)
            bbox = np.array([y, x, y+h, x+h], dtype=np.int32).T
            assert ~np.any(np.isnan(x)), "FOUND NAN"
            assert ~np.any(np.isnan(y)), "FOUND NAN"
            assert ~np.any(np.isnan(h)), "FOUND NAN"
            assert ~np.any(np.isnan(bbox)), "FOUND NAN"
            bbox = np.concatenate((bbox, [[0, 0, 1, 1]]))
            class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
            if self.augmentation:
                image, bbox, class_ids = self.augment_data(
                    image, bbox, class_ids)
                input_images.extend(image)
                input_bboxes.extend(bbox)
                input_class_ids.extend(class_ids)
            else:
                input_images.append(image)
                input_bboxes.append(bbox)
                input_class_ids.append(class_ids)
        input_images = np.array(input_images, dtype=np.float32)
        input_images = self.normalize_images_with_lim(input_images)
        assert ~np.any(np.isnan(input_images)), "FOUND NAN"
        return input_images, input_bboxes, input_class_ids

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "resid":
            return info["object"]
        else:
            super(self.__class__).image_reference(self, image_id)


class Resid_btk_model(object):
    def __init__(self, model_name, model_path, output_dir,
                 training=False, images_per_gpu=1,
                 validation_for_training=False,
                 *args, **kwargs):
        self.model_name = model_name
        self.training = training
        self.model_path = model_path
        self.output_dir = output_dir
        self.validation_for_training = validation_for_training

        class InferenceConfig(mrcnn.config_btk_only.Config):
            NAME = self.model_name
            GPU_COUNT = 1
            IMAGES_PER_GPU = images_per_gpu
            STEPS_PER_EPOCH = 500  # 200
            VALIDATION_STEPS = 20
            RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            DETECTION_MIN_CONFIDENCE = 0.965  # 0.95

        self.config = InferenceConfig()
        if self.training:
            self.config.display()
        else:
            self.config.TRAIN_BN = False

    def make_resid_model(self, train_catalog_name, count=256,
                         sampling_function=None, max_number=2,
                         augmentation=False, norm_val=None,
                         selection_function=None, train_wld_catalog_name=None,
                         meas_params=None, input_pull=False,
                         input_model_mapping=False, obs_condition=None,
                         val_wld_catalog_name=None, val_catalog_name=None,
                         multiprocess=False):
        """Creates dataset and loads model"""
        # If no user input sampling function then set default function
        import mrcnn.model_btk_only as model_btk
        if not sampling_function:
            sampling_function = resid_general_sampling_function
        train_wld_catalog = astropy.table.Table.read(
            train_wld_catalog_name, format='fits')
        self.meas_generator = make_meas_generator(train_catalog_name,
                                                  self.config.BATCH_SIZE,
                                                  max_number,
                                                  sampling_function,
                                                  selection_function,
                                                  train_wld_catalog,
                                                  meas_params,
                                                  obs_condition,
                                                  multiprocess)
        self.dataset = ResidDataset(self.meas_generator, norm_val=norm_val,
                                    augmentation=augmentation,
                                    input_pull=input_pull,
                                    input_model_mapping=input_model_mapping)
        self.dataset.load_data(count=count)
        self.dataset.prepare()
        if augmentation:
            self.config.BATCH_SIZE *= 4
            self.config.IMAGES_PER_GPU *= 4
        if self.training:
            self.model = model_btk.MaskRCNN(mode="training",
                                            config=self.config,
                                            model_dir=self.output_dir)
            if self.validation_for_training:
                val_wld_catalog = astropy.table.Table.read(
                    val_wld_catalog_name, format='fits')
                val_meas_generator = make_meas_generator(val_catalog_name,
                                                         self.config.BATCH_SIZE,
                                                         max_number,
                                                         sampling_function,
                                                         selection_function,
                                                         val_wld_catalog,
                                                         meas_params,
                                                         obs_condition,
                                                         multiprocess)
                self.dataset_val = ResidDataset(
                    val_meas_generator, norm_val=norm_val,
                    input_pull=input_pull,
                    input_model_mapping=input_model_mapping)
                self.dataset_val.load_data(count=count)
                self.dataset_val.prepare()
        else:
            print("detection minimum confidence score:",
                  self.config.DETECTION_MIN_CONFIDENCE)
            self.model = model_btk.MaskRCNN(mode="inference",
                                            config=self.config,
                                            model_dir=self.output_dir)
        if self.model_path:
            print("Loading weights from ", self.model_path)
            self.model.load_weights(self.model_path, by_name=True)


class Resid_btk_model_gold(object):
    def __init__(self, model_name, model_path, output_dir,
                 training=False, images_per_gpu=1,
                 validation_for_training=False, *args, **kwargs):
        self.model_name = model_name
        self.training = training
        self.model_path = model_path
        self.output_dir = output_dir
        self.validation_for_training = validation_for_training

        class InferenceConfig(mrcnn.config_btk_only.Config):
            NAME = self.model_name
            GPU_COUNT = 1
            IMAGES_PER_GPU = images_per_gpu
            STEPS_PER_EPOCH = 500  # 200
            VALIDATION_STEPS = 20
            RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            DETECTION_MIN_CONFIDENCE = 0.965  # 0.95

    def make_resid_model_gold(self, catalog_name, count=256,
                              sampling_function=None, max_number=2,
                              augmentation=False, norm_val=None,
                              selection_function=None, wld_catalog_name=None,
                              meas_params=None, multiprocess=False):
        """Creates dataset and loads model"""
        # If no user input sampling function then set default function
        import mrcnn.model_btk_only as model_btk
        if not sampling_function:
            sampling_function = resid_general_sampling_function
        self.meas_generator = make_meas_generator(catalog_name,
                                                  self.config.BATCH_SIZE,
                                                  max_number,
                                                  sampling_function,
                                                  selection_function,
                                                  wld_catalog_name,
                                                  meas_params)
        self.dataset = ResidDataset(self.meas_generator, norm_val=norm_val,
                                    augmentation=augmentation, i_mag_lim=25.3)
        self.dataset.load_data(count=count)
        self.dataset.prepare()
        if augmentation:
            self.config.BATCH_SIZE *= 4
            self.config.IMAGES_PER_GPU *= 4
        if self.training:
            self.model = model_btk.MaskRCNN(mode="training",
                                            config=self.config,
                                            model_dir=self.output_dir)
            if self.validation_for_training:
                val_meas_generator = make_meas_generator(catalog_name,
                                                         self.config.BATCH_SIZE,
                                                         max_number,
                                                         sampling_function,
                                                         selection_function,
                                                         wld_catalog_name,
                                                         meas_params)
                self.dataset_val = ResidDataset(val_meas_generator,
                                                norm_val=norm_val,
                                                i_mag_lim=25.3)
                self.dataset_val.load_data(count=count)
                self.dataset_val.prepare()
        else:
            print("detection minimum confidence score:",
                  self.config.DETECTION_MIN_CONFIDENCE)
            self.model = model_btk.MaskRCNN(mode="inference",
                                            config=self.config,
                                            model_dir=self.output_dir)
        if self.model_path:
            print("Loading weights from ", self.model_path)
            self.model.load_weights(self.model_path, by_name=True)


def stack_resid_merge_centers(det_cent, resid_cent,
                              distance_upper_bound=1):
    """Combines centers detected by stack on image and in iterative step.
    """
    # remove duplicates
    if len(resid_cent) == 0:
        if len(det_cent) == 0:
            return []
        else:
            return np.unique(det_cent, axis=0)
    if len(det_cent) == 0:
        return resid_cent
    unique_det_cent = np.unique(det_cent, axis=0)
    z_tree = spatial.KDTree(unique_det_cent)
    resid_cent = resid_cent.reshape(-1, 2)
    match = z_tree.query(resid_cent,
                         distance_upper_bound=distance_upper_bound)
    trim = np.setdiff1d(range(len(unique_det_cent)), match[1])
    trim_det_cent = [unique_det_cent[i] for i in trim]
    if len(trim_det_cent) == 0:
        return resid_cent
    iter_det = np.vstack([trim_det_cent, resid_cent])
    return iter_det


class Stack_iter_btk_param(btk.compute_metrics.Metrics_params):
    def __init__(self, catalog_name, batch_size=1, max_number=2,
                 sampling_function=None, selection_function=None,
                 wld_catalog_name=None, meas_params=None, detect_coadd=False,
                 *args, **kwargs):
        super(Stack_iter_btk_param, self).__init__(*args, **kwargs)
        if not sampling_function:
            print("resid_sampling")
            sampling_function = resid_general_sampling_function
        self.meas_generator = make_meas_generator(
            catalog_name, batch_size, max_number, sampling_function,
            selection_function, wld_catalog_name, meas_params)
        self.detect_coadd = detect_coadd

    def get_resid_iter_detections(self, index):
        """
        Returns model detected centers and true center for data entry index.
        Args:
            index: Index of dataset to perform detection on.
        Returns:
            x and y coordinates of iteratively detected centers, centers
            detected initially and true centers.
        Useful for evaluating model detection performance."""
        image, gt_bbox, gt_class_id = self.dataset.load_input()
        true_centers = self.dataset.true_cent
        detected_centers = self.dataset.det_cent
        results1 = self.model.detect(image, verbose=0)
        iter_detected_centers = []
        for i, r1 in enumerate(results1):
            iter_detected_centers.append(resid_merge_centers(
                detected_centers[i], r1['rois'], center_shift=0))
        return iter_detected_centers, detected_centers, true_centers

    def get_iter_centers(self):
        """Performs stack detection on residual image and returns detected
        centroids."""
        output, deb, cat = next(self.meas_generator)
        self.output = output
        self.deblend_output = deb
        self.obs_cond = output['obs_condition']
        resid_centers = []
        self.det_cent, self.true_cent = [], []
        for i in range(len(output['blend_list'])):
            blend_image = output['blend_images'][i]
            blend_list = output['blend_list'][i]
            model_image = deb[i][0]
            detected_centers = deb[i][1]
            self.det_cent.append(detected_centers)
            cent_t = [np.array(blend_list['dx']), np.array(blend_list['dy'])]
            self.true_cent.append(np.stack(cent_t).T)
            resid_images = blend_image - model_image
            resid_cat = get_stack_catalog(resid_images,
                                          output['obs_condition'][i],
                                          detect_coadd=self.detect_coadd)
            resid_centers.append(get_stack_centers(resid_cat))
        return resid_centers

    def get_stck_iter_detections(self, index):
        """Returns model detected centers and true center for data entry index.
        Args:
            index: Index of dataset to perform detection on.
        Returns:
            x and y coordinates of iteratively detected centers, centers
            detected initially and true centers.
        Useful for evaluating model detection performance."""
        results1 = self.get_iter_centers()
        iter_detected_centers = []
        for i, r1 in enumerate(results1):
            iter_detected_centers.append(stack_resid_merge_centers(
                self.det_cent[i], r1))
        return iter_detected_centers
