import sep
import sys
import os
import numpy as np
BTK_PATH = '/home/users/sowmyak/BlendingToolKit/'
sys.path.insert(0, BTK_PATH)
import btk
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)
import btk_utils
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/resid/data'


def run_stack(image_array, variance_array, psf_array,
              min_pix=1, bkg_bin_size=32, thr_value=5):
    """
    Function to setup the DM stack and perform detection, deblending and
    measurement
    Args:
        image_array: Numpy array of image to run stack on
        variance_array: per pixel variance of the input image_array (must
                        have same dimensions as image_array)
        psf_array: Image of the PSF for image_array.
        min_pix: Minimum size in pixels of a source to be considered by the
                 stack (default=1).
        bkg_bin_size: Binning of the local background in pixels (default=32).
        thr_value: SNR threshold for the detected sources to be included in the
                   final catalog(default=5).
    Returns:
        catalog: AstroPy table of detected sources
    """
    # Convert to stack Image object
    import lsst.afw.table
    import lsst.afw.image
    import lsst.afw.math
    import lsst.meas.algorithms
    import lsst.meas.base
    import lsst.meas.deblender
    import lsst.meas.extensions.shapeHSM
    image = lsst.afw.image.ImageF(image_array)
    variance = lsst.afw.image.ImageF(variance_array)
    # Generate a masked image, i.e., an image+mask+variance image (mask=None)
    masked_image = lsst.afw.image.MaskedImageF(image, None, variance)
    # Create the kernel in the stack's format
    psf_im = lsst.afw.image.ImageD(psf_array)
    fkernel = lsst.afw.math.FixedKernel(psf_im)
    psf = lsst.meas.algorithms.KernelPsf(fkernel)
    # Passing the image to the stack
    exposure = lsst.afw.image.ExposureF(masked_image)
    # Assign the exposure the PSF that we created
    exposure.setPsf(psf)
    schema = lsst.afw.table.SourceTable.makeMinimalSchema()
    config1 = lsst.meas.algorithms.SourceDetectionConfig()
    # Tweaks in the configuration that can improve detection
    # Change carefully!
    #####
    config1.tempLocalBackground.binSize = bkg_bin_size
    config1.minPixels = min_pix
    config1.thresholdValue = thr_value
    #####
    detect = lsst.meas.algorithms.SourceDetectionTask(schema=schema,
                                                      config=config1)
    deblend = lsst.meas.deblender.SourceDeblendTask(schema=schema)
    config1 = lsst.meas.base.SingleFrameMeasurementConfig()
    config1.plugins.names.add('ext_shapeHSM_HsmShapeRegauss')
    config1.plugins.names.add('ext_shapeHSM_HsmSourceMoments')
    config1.plugins.names.add('ext_shapeHSM_HsmPsfMoments')
    measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema,
                                                        config=config1)
    table = lsst.afw.table.SourceTable.make(schema)
    detect_result = detect.run(table, exposure)  # run detection task
    catalog = detect_result.sources
    deblend.run(exposure, catalog)  # run the deblending task
    measure.run(catalog, exposure)  # run the measuring task
    catalog = catalog.copy(deep=True)


class Stack_params(btk.measure.Measurement_params):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5
    psf_stamp_size = 41

    def get_psf_sky(self, obs_cond):
        mean_sky_level = obs_cond.mean_sky_level
        psf = obs_cond.psf_model
        psf_image = psf.drawImage(
           nx=self.psf_stamp_size,
           ny=self.psf_stamp_size).array
        return psf_image, mean_sky_level

    def make_measurement(self, data, index):
        """Perform detection, deblending and measurement on the i band image of
        the blend image for input index in the batch.
         """
        image_array = data['blend_images'][index, :, :, 3].astype(np.float32)
        psf_image, mean_sky_level = self.get_psf_sky(data['obs_condition'][3])
        variance_array = image_array + mean_sky_level
        psf_array = psf_image.astype(np.float64)
        cat = run_stack(image_array, variance_array, psf_array,
                        min_pix=self.min_pix, bkg_bin_size=self.bkg_bin_size,
                        thr_value=self.thr_value)
        cat_chldrn = cat[cat['deblend_nChild'] == 0]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()


class Stack_params_i_band(btk.measure.Measurement_params):
    """Class containing functions to perform detection"""
    def get_centers(self, image):
        """Returns x and y coordinates of object centroids detected by sep.
        Detection is performed in the i band.
        Args:
            image: multi-band image to perform detection on. ([#bands, x, y])
        Returns:
            x and y coordinates of detected centroids.
        """
        detect = image[3]  # detection in i band
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        return np.stack((catalog['x'], catalog['y']), axis=1)

    def get_deblended_images(self, data, index):
        """Returns detected centers for the given blend
        Args:
            data: output from btk.draw_blends generator
            index: index of blend in bacth_outputs to perform analysis on.
        Returns:
            deblended images and detected centers
        """
        image = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        peaks = self.get_centers(image)
        return [None, peaks]


def get_btk_generator(meas_params, max_number, sampling_function,
                      selection_function, wld_catalog):
    """Returns btk.measure generator for input settings

    """
    # Input catalog name
    catalog_name = os.path.join("/scratch/users/sowmyak/data", 'OneDegSq.fits')
    # Load parameters
    param = btk.config.Simulation_params(
        catalog_name, max_number=max_number, batch_size=8, stamp_size=25.6)
    if wld_catalog:
            param.wld_catalog = wld_catalog
    np.random.seed(param.seed)
    # Load input catalog
    catalog = btk.get_input_catalog.load_catlog(
        param, selection_function=selection_function)
    # Generate catalogs of blended objects
    blend_generator = btk.create_blend_generator.generate(
        param, catalog, sampling_function)
    # Generates observing conditions for the selected survey_name & all bands
    observing_generator = btk.create_observing_generator.generate(
        param, btk_utils.resid_obs_conditions)
    # Generate images of blends in all the observing bands
    draw_blend_generator = btk.draw_blends.generate(
        param, blend_generator, observing_generator)
    meas_generator = btk.measure.generate(
        meas_params, draw_blend_generator, param)
    return meas_generator


def run_sep(save_file_name, test_size, meas_params, max_number,
            sampling_function, selection_function=None, wld_catalog=None):
    """Test performance for btk input blends"""
    meas_generator = get_btk_generator(
        max_number, sampling_function, selection_function, wld_catalog)
    results = []
    np.random.seed(1)
    for im_id in range(test_size):
        output, deb, _ = next(meas_generator)
        blend_list = output['blend_list'][0]
        detected_centers = deb[0][1]
        true_centers = np.stack([blend_list['dx'], blend_list['dy']]).T
        det, undet, spur = btk.compute_metrics.evaluate_detection(
            detected_centers, true_centers)
        print(len(true_centers), det, undet, spur)
        results.append([len(true_centers), det, undet, spur])
    arr_results = np.array(results).T
    print("Results: ", np.sum(arr_results, axis=1))
    np.savetxt(save_file_name, arr_results)


if __name__ == '__main__':
    max_number = 2
    test_size = 15
    # Run sep coadd detection
    meas_params = Stack_params
    save_file_name = f"stack_2gal_coadd_results.txt"
    run_sep(save_file_name, test_size, meas_params, max_number)
    # Run sep i band detection
    meas_params = Stack_params_i_band
    save_file_name = f"stack_2gal_iband_results.txt"
    run_sep(save_file_name, test_size, meas_params, max_number)
