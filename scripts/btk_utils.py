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
from functools import partial


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


def group_sampling_function(Args, catalog):
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
    # randomly sample a group.
    group_ids = np.unique(
        Args.wld_catalog['grp_id'][
            (Args.wld_catalog['grp_size'] >= 2) &
            (Args.wld_catalog['grp_size'] <= Args.max_number)])
    group_id = np.random.choice(group_ids)
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
               f"blend. Found {len(no_boundary)}, "
               f"expected <= {Args.max_number}")
    assert len(no_boundary) <= Args.max_number, message
    return no_boundary


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
    if not hasattr(Args, 'wld_catalog_name'):
        raise Exception(
            "A pre-run WLD catalog should be input as Args.wld_catalog_name")
    wld_catalog = astropy.table.Table.read(
        Args.wld_catalog_name, format='fits')
    if not hasattr(Args, 'group_id_count'):
        raise NameError("An integer specifying index of group_id to draw must"
                        "be input as Args.group_id_count")
    elif not isinstance(Args.group_id_count, int):
        raise ValueError("group_id_count must be an integer")
    # randomly sample a group.
    group_ids = np.unique(
        wld_catalog['grp_id'][
            (wld_catalog['grp_size'] >= 2) &
            (wld_catalog['grp_size'] <= Args.max_number)])
    if Args.group_id_count >= len(group_ids):
        message = (f"group_id_count:{Args.group_id_count} is larger than the"
                   f"number of groups input {len(group_ids)}")
        raise GeneratorExit(message)
    else:
        group_id = group_ids[Args.group_id_count]
        Args.group_id_count += 1
    # get all galaxies belonging to the group.
    # make sure some group or galaxy was not repeated in wld_catalog
    ids = np.unique(
        wld_catalog['db_id'][wld_catalog['grp_id'] == group_id])
    temp_catalog = astropy.table.vstack(
        [catalog[catalog['galtileid'] == i] for i in ids])[:Args.max_number]
    blend_catalog = astropy.table.unique(temp_catalog, keys='galtileid')
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
    except (np.linalg.LinAlgError, ValueError, ArithmeticError) as e:
        print("multi component initialization failed \n", e)
        blend, observation = scarlet1_initialize(
            images, peaks, psfs, variances, np.array(bands, dtype=str))
        try:
            blend.fit(iters, e_rel=e_rel, f_rel=f_rel)
            scarlet_multi_fit = 0
        except(np.linalg.LinAlgError, ValueError, ArithmeticError) as e:
            print("scarlet did not fit \n", e)
            scarlet_multi_fit = -1
    return blend, observation, scarlet_multi_fit


class Scarlet_resid_params(object):
    def __init__(self, iters=200, e_rel=.015, f_rel=1e-6,
                 detect_centers=True, detect_coadd=False,
                 *args, **kwargs):
        # super(Scarlet_resid_params, self).__init__(*args, **kwargs)
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
        except(ValueError) as e:
            print("Unable to create scarlet model \n", e)
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
        psf_image = np.zeros([psf_stamp_size, psf_stamp_size])
        for i in range(len(obs_cond)):
            psf, mean_sky_level = get_psf_sky(obs_cond[i],
                                              psf_stamp_size)
            variance_image += image[:, :, i] + mean_sky_level
            input_image += image[:, :, i]
            psf_image += psf
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


class Stack_iter_params(object):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5
    psf_stamp_size = 41
    iters = 200
    e_rel = .015
    f_rel = 1e-6

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
        catalog = get_stack_catalog(data['blend_images'][index],
                                    data['obs_condition'][index],
                                    detect_coadd=self.detect_coadd,
                                    psf_stamp_size=self.psf_stamp_size,
                                    min_pix=self.min_pix,
                                    bkg_bin_size=self.bkg_bin_size,
                                    thr_value=self.thr_value)
        self.catalog[index] = catalog
        peaks = get_stack_centers(catalog)
        #import ipdb;ipdb.set_trace()
        if len(peaks) == 0:
            print("Unable to create scarlet model, no peaks")
            temp_model = np.zeros_like(data['blend_images'][index])
            return {'scarlet_model': temp_model, 'scarlet_peaks': [],
                    'scarlet_multi_fit': 0}
        try:
            blend, observation, sf = scarlet_fit(
                images, peaks, psfs, variances, bands,
                self.iters, self.e_rel, self.f_rel)
            temp_model = np.zeros_like(data['blend_images'][index])
            selected_peaks = []
            for k, component in enumerate(blend):
                y, x = component.center
                selected_peaks.append([x, y])
                model = component.get_model()
                model_ = observation.render(model)
                temp_model += np.transpose(model_, axes=(1, 2, 0))
        except(ValueError) as e:
            print("Unable to create scarlet model \n", e)
            return {'scarlet_model': temp_model, 'scarlet_peaks': [],
                    'scarlet_multi_fit': sf}
        return {'scarlet_model': temp_model, 'scarlet_peaks': selected_peaks,
                'scarlet_multi_fit': sf}


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
            print("Setting scarlet_resid_params as meas_params")
            meas_params = Scarlet_resid_params()
        if multiprocess:
            cpus = 4  # multiprocessing.cpu_count()
        else:
            cpus = 1
        meas_generator = btk.measure.generate(
            meas_params, draw_blend_generator, param,
            multiprocessing=multiprocess, cpus=cpus)
        return meas_generator


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


class SEP_i_band_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self):
        self.detect_coadd = False

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        images[np.isnan(images)] = 0
        if self.detect_coadd:
            peaks = self.get_centers_coadd(images)
        else:
            peaks = self.get_centers_i_band(images)
        return {'deblend_image': None, 'peaks': peaks}


class SEP_coadd_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self):
        self.detect_coadd = True

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        images[np.isnan(images)] = 0
        if self.detect_coadd:
            peaks = self.get_centers_coadd(images)
        else:
            peaks = self.get_centers_i_band(images)
        return {'deblend_image': None, 'peaks': peaks}


class Stack_iter_i_band_measure_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self, verbose=False):
        self.scarlet_param = Stack_iter_params(detect_coadd=False)
        self.iter_stack = Stack_iter_params(detect_coadd=False)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        s_mf = scarlet_op['scarlet_multi_fit']
        self.det_cent = detected_centers
        cent_t = [np.array(blend_list['dx']), np.array(blend_list['dy'])]
        self.true_cent = np.stack(cent_t).T
        resid_image = blend_image - model_image
        iter_data = {'blend_images': [resid_image, ],
                     'obs_condition': [obs_cond, ]}
        stck_peaks = self.iter_stack.get_only_peaks(
            iter_data, 0)
        iter_peaks = stack_resid_merge_centers(self.det_cent,
                                               stck_peaks)
        if len(iter_peaks) == 0:
            iter_peaks = np.empty((0, 2))
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': iter_peaks, 'scarlet_mf': s_mf}

    def make_measurement(self, data=None, index=None):
        """Function describing how the measurement algorithm is run.
        Note this resturns catalog from primary detection.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                         images, isolated images, observing conditions and
                         blend catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                         measurement on.

        Returns:
            output of measurement algorithm as a dict.
        """
        cat = self.scarlet_param.catalog[index]
        cat_select = cat[
            (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
        return cat_select


class Stack_iter_measure_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self, verbose=False):
        self.scarlet_param = Stack_iter_params(detect_coadd=True)
        self.iter_stack = Stack_iter_params(detect_coadd=True)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        s_mf = scarlet_op['scarlet_multi_fit']
        self.det_cent = detected_centers
        cent_t = [np.array(blend_list['dx']), np.array(blend_list['dy'])]
        self.true_cent = np.stack(cent_t).T
        resid_image = blend_image - model_image
        iter_data = {'blend_images': [resid_image, ],
                     'obs_condition': [obs_cond, ]}
        stack_peaks = self.iter_stack.get_only_peaks(
            iter_data, 0)
        iter_peaks = stack_resid_merge_centers(self.det_cent,
                                               stack_peaks)
        if len(iter_peaks) == 0:
            iter_peaks = np.empty((0, 2))
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': iter_peaks, 'scarlet_mf': s_mf}

    def make_measurement(self, data=None, index=None):
        """Function describing how the measurement algorithm is run.
        Note this resturns catalog from primary detection.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                         images, isolated images, observing conditions and
                         blend catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                         measurement on.

        Returns:
            output of measurement algorithm as a dict.
        """
        cat = self.scarlet_param.catalog[index]
        cat_select = cat[
            (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
        return cat_select


class Stack_coadd_params(btk.measure.Measurement_params):
    """Class with functions that describe how LSST science pipeline can
    perform measurements on the input data."""
    min_pix = 1  # Minimum size in pixels to be considered a source
    bkg_bin_size = 32  # Binning size of the local background
    thr_value = 5  # SNR threshold for the detection
    psf_stamp_size = 41  # size of pstamp to draw PSF on
    detect_coadd = True

    def make_measurement(self, data, index):
        """Perform detection, deblending and measurement on the i band image of
        the blend for input index entry in the batch.

        Args:
            data: Dictionary with blend images, isolated object images, blend
                catalog, and observing conditions.
            index: Position of the blend to measure in the batch.

        Returns:
            astropy.Table of the measurement results.
         """
        image = data['blend_images'][index]
        obs_cond = data['obs_condition'][index]
        image_array, variance_array, psf_image = get_stack_input(
            image, obs_cond, self.psf_stamp_size, self.detect_coadd)
        psf_array = psf_image.astype(np.float64)
        cat = btk.utils.run_stack(
            image_array, variance_array, psf_array, min_pix=self.min_pix,
            bkg_bin_size=self.bkg_bin_size, thr_value=self.thr_value)
        cat_chldrn = cat[
            (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()

    def get_deblended_images(self, data=None, index=None):
        return None


class Stack_i_band_params(btk.measure.Measurement_params):
    """Class with functions that describe how LSST science pipeline can
    perform measurements on the input data."""
    min_pix = 1  # Minimum size in pixels to be considered a source
    bkg_bin_size = 32  # Binning size of the local background
    thr_value = 5  # SNR threshold for the detection
    psf_stamp_size = 41  # size of pstamp to draw PSF on
    detect_coadd = False

    def make_measurement(self, data, index):
        """Perform detection, deblending and measurement on the i band image of
        the blend for input index entry in the batch.

        Args:
            data: Dictionary with blend images, isolated object images, blend
                catalog, and observing conditions.
            index: Position of the blend to measure in the batch.

        Returns:
            astropy.Table of the measurement results.
         """
        image = data['blend_images'][index]
        obs_cond = data['obs_condition'][index]
        image_array, variance_array, psf_image = get_stack_input(
            image, obs_cond, self.psf_stamp_size, self.detect_coadd)
        psf_array = psf_image.astype(np.float64)
        cat = btk.utils.run_stack(
            image_array, variance_array, psf_array, min_pix=self.min_pix,
            bkg_bin_size=self.bkg_bin_size, thr_value=self.thr_value)
        cat_chldrn = cat[
            (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()

    def get_deblended_images(self, data=None, index=None):
        return None


class Stack_metric_params(btk.compute_metrics.Metrics_params):
    def __init__(self, *args, **kwargs):
        super(Stack_metric_params, self).__init__(*args, **kwargs)
        """Class describing functions to return results of
        detection/deblending/measurement algorithm in meas_generator.  Each
        time the algorithm is called, it is run on a batch of blends yielded
        by the meas_generator.
        """

    def get_detections(self):
        """Returns blend catalog and detection catalog for detection performed.

        Returns:
            Results of the detection algorithm are returned as:
                true_tables: List of astropy Table of the blend catalogs of the
                    batch. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
                detected_tables: List of astropy Table of output from detection
                    algorithm. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
        """
        blend_op, _, cat = next(self.meas_generator)
        # Get astropy table with entries corresponding to true sources
        true_tables = blend_op['blend_list']
        detected_tables = []
        for i in range(len(true_tables)):
            detected_centers = np.stack(
                [cat[i]['base_SdssCentroid_x'],
                 cat[i]['base_SdssCentroid_y']],
                axis=1)
            detected_table = astropy.table.Table(detected_centers,
                                                 names=['dx', 'dy'])
            detected_tables.append(detected_table)
        return true_tables, detected_tables
