import btk
import os
import numpy as np
import astropy.table
import sys

sys.path.append('..')
sys.path.append('../..')
sys.path.append('scripts')
sys.path.append('/home/users/sowmyak/ResidualDetectron/scripts/')
import btk_utils
import mrcnn.model_btk_only as model_btk


MODEL_DIR = '/scratch/users/sowmyak/resid/logs_oct'
#model_file_name = '3resid_btk_square_again20190213T1020/mask_rcnn_3resid_btk_square_again_0225.h5'
# #model_file_name = '3resid_btk_square_again20190213T1020/mask_rcnn_3resid_btk_square_again_0244.h5'
# model_file_name1 = 'resid_btk_square_group_3_20190213T1020/mask_rcnn_resid_btk_square_group_3__0263.h5'
# model_file = 'model12_again220191010T1740/mask_rcnn_model12_again2_0472.h5'
model_file = 'model12_again220191010T1740/mask_rcnn_model12_again2_0481.h5'
model_file_name = os.path.join(MODEL_DIR, model_file)
model_name = '15_8'
DATA_PATH = '/scratch/users/sowmyak/data'
BATCH_SIZE = 1


class RD_measure_params_temp(btk.measure.Measurement_params):
    def __init__(self, model_file_name=model_file_name,
                 model_name=model_name, verbose=False):
        self.model_file_name = model_file_name
        self.model_name = model_name
        self.get_model(verbose=False)
        self.scarlet_param = btk_utils.Scarlet_resid_params()
        self.norm = [0., 1.45, 0, 1.]

    def get_model(self, verbose=False):
        # catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
        resid_model = btk_utils.Resid_btk_model(
            self.model_name, self.model_file_name, MODEL_DIR, training=False,
            images_per_gpu=BATCH_SIZE)
        # Load parameters for dataset and load model
        resid_model.config.WEIGHT_DECAY = 0.001
        resid_model.config.STEPS_PER_EPOCH = 1000
        resid_model.config.VALIDATION_STEPS = 20
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        print("Evaluate model:", self.model_name)
        resid_model.config.display()
        # resid_model.make_resid_model(catalog_name, count=BATCH_SIZE)
        resid_model.model = model_btk.MaskRCNN(mode="inference",
                                               config=resid_model.config,
                                               model_dir=resid_model.output_dir)
        resid_model.model.load_weights(self.model_file_name, by_name=True)
        self.resid_model = resid_model

    def normalize_images(self, images):
        images[:, :, 0:6] = (images[:, :, 0:6] - self.norm[0])/self.norm[1]
        images[:, :, 6:12] = (images[:, :, 6:12] - self.norm[2])/self.norm[3]
        return images

    def get_deblended_images(self, data, index):
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        self.det_cent = detected_centers
        self.true_cent = np.stack([blend_list['dx'], blend_list['dy']]).T
        resid_image = blend_image - model_image
        bg_rms = [c.mean_sky_level for c in obs_cond]
        resid_image /= np.sqrt(np.array(bg_rms) + blend_image)
        stretch = 2000  # 0.1
        Q = 0.5  # 3
        model_image = np.arcsinh(Q*model_image/stretch)/Q
        image = np.dstack([resid_image, model_image])
        x, y, h = btk_utils.get_undetected(
            blend_list, detected_centers, obs_cond[3])
        bbox = np.array([y, x, y+h, x+h], dtype=np.int32).T
        assert ~np.any(np.isnan(x)), "FOUND NAN"
        assert ~np.any(np.isnan(y)), "FOUND NAN"
        assert ~np.any(np.isnan(h)), "FOUND NAN"
        assert ~np.any(np.isnan(bbox)), "FOUND NAN"
        bbox = np.concatenate((bbox, [[0, 0, 1, 1]]))
        class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
        input_images = np.array(image, dtype=np.float32)
        input_images = self.normalize_images(input_images)
        resid_result = self.resid_model.model.detect([input_images], verbose=0)[0]
        it_det_cent = btk_utils.resid_merge_centers(self.det_cent,
                                                    resid_result['rois'])
        if len(it_det_cent) == 0:
            it_det_cent = np.empty((0, 2))
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': it_det_cent, 'prediction': resid_result,
                'target_bbox': bbox, 'target_class': class_ids
                }


class RD_2gal_measure_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""
    def __init__(self, model_file_name=model_file_name,
                 verbose=False):
        self.model_file_name = model_file_name
        self.get_model(verbose=False)
        self.scarlet_param = btk_utils.Scarlet_resid_params(detect_coadd=True)
        self.iter_scarlet_param = btk_utils.Scarlet_resid_params(
            detect_centers=False)

    def get_model(self, verbose=False):
        """Return centers detected when object detection and photometry
        is done on input image with SEP.
        Args:
            image: Image (single band) of galaxy to perform measurement on.
        Returns:
                centers: x and y coordinates of detected  centroids

        """
        # set detection threshold to 5 times std of image
        self.model_name = model_name
        self.model_path = os.path.join(MODEL_DIR, self.model_file_name)
        self.norm = [1.9844158727667542, 413.83759806375525,
                     51.2789974336363, 1038.4760551905683]
        #[1.6361405416087091, 416.16687641284665, 63.16814480535191, 2346.133101333463]
        count = 4000
        catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
        resid_model = btk_utils.Resid_btk_model(self.model_name, self.model_path,
                                                MODEL_DIR, training=False,
                                                images_per_gpu=1)
        if verbose:
            resid_model.config.display()
        resid_model.make_resid_model(catalog_name, count=count,
                                     max_number=2, norm_val=self.norm)
        self.resid_model = resid_model

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        self.det_cent = detected_centers
        self.true_cent = np.stack([blend_list['dx'], blend_list['dy']]).T
        resid_image = blend_image - model_image
        image = np.dstack([resid_image, model_image])
        x, y, h = btk_utils.get_undetected(
            blend_list, detected_centers, obs_cond[3])
        bbox = np.array([y, x, y+h, x+h], dtype=np.int32).T
        assert ~np.any(np.isnan(x)), "FOUND NAN"
        assert ~np.any(np.isnan(y)), "FOUND NAN"
        assert ~np.any(np.isnan(h)), "FOUND NAN"
        assert ~np.any(np.isnan(bbox)), "FOUND NAN"
        bbox = np.concatenate((bbox, [[0, 0, 1, 1]]))
        class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
        input_images = np.array(image, dtype=np.float32)
        input_images = self.normalize_images(input_images)
        resid_result = self.resid_model.model.detect([input_images], verbose=0)[0]
        it_det_cent = btk_utils.resid_merge_centers(self.det_cent,
                                                    resid_result['rois'])
        if len(it_det_cent) == 0:
            it_det_cent = np.empty((0, 2))
        #iter_data = {'blend_images': data['blend_images'],
        #             'obs_condition': data['obs_condition']}
        #iter_scarlet_op = self.iter_scarlet_param.get_deblended_images(
        #    iter_data, index, peaks=it_det_cent)
        #scarlet_peaks = iter_scarlet_op['scarlet_peaks']
        #iter_deblend_image = iter_scarlet_op['scarlet_model']
        #peaks = np.unique(scarlet_peaks, axis=0)
        #return {'deblend_image': model_image, 'resid_image': resid_image,
        #        'it_det_cent': it_det_cent, 'iter_deblend_image': iter_deblend_image,
        #        'peaks': peaks, 'prediction': resid_result,
        #        'target_bbox': bbox, 'target_class': class_ids
        #        }
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': it_det_cent, 'prediction': resid_result,
                'target_bbox': bbox, 'target_class': class_ids
                }

    def normalize_images(self, images):
        images[:, :, 0:6] = (images[:, :, 0:6] - self.norm[0])/self.norm[1]
        images[:, :, 6:12] = (images[:, :, 6:12] - self.norm[2])/self.norm[3]
        return images


class RD_group_measure_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""
    def __init__(self, model_file_name=model_file_name,
                 verbose=False, i_mag_lim=30):
        self.i_mag_lim = i_mag_lim
        self.model_file_name = model_file_name
        self.get_model(verbose=False)
        self.scarlet_param = btk_utils.Scarlet_resid_params(detect_coadd=True)
        self.iter_scarlet_param = btk_utils.Scarlet_resid_params(
            detect_centers=False)

    def get_model(self, verbose=False, DETECTION_MIN_CONFIDENCE=0.96):
        """Return centers detected when object detection and photometry
        is done on input image with SEP.
        Args:
            image: Image (single band) of galaxy to perform measurement on.
        Returns:
                centers: x and y coordinates of detected  centroids

        """
        # set detection threshold to 5 times std of image
        self.model_name = model_name
        self.model_path = os.path.join(MODEL_DIR, self.model_file_name)
        self.norm = [1.9844158727667542, 413.83759806375525,
                     51.2789974336363, 1038.4760551905683]
        #[1.6361405416087091, 416.16687641284665, 63.16814480535191, 2346.133101333463]
        count = 4000
        catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
        resid_model = btk_utils.Resid_btk_model(self.model_name, self.model_path,
                                                MODEL_DIR, training=False,
                                                images_per_gpu=1)
        resid_model.config.DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE
        if verbose:
            resid_model.config.display()
        wld_catalog_name = os.path.join(DATA_PATH, 'train_group_min_snr_01.fits')
        resid_model.make_resid_model(
            catalog_name, count=count, max_number=6, norm_val=self.norm,
            sampling_function=btk.utils.group_sampling_function,
            wld_catalog_name=wld_catalog_name)
        self.resid_model = resid_model

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        self.det_cent = detected_centers
        self.true_cent = np.stack([blend_list['dx'], blend_list['dy']]).T
        resid_image = blend_image - model_image
        image = np.dstack([resid_image, model_image])
        x, y, h = btk_utils.get_undetected(
            blend_list, detected_centers, obs_cond[3], self.i_mag_lim)
        bbox = np.array([y, x, y+h, x+h], dtype=np.int32).T
        assert ~np.any(np.isnan(x)), "FOUND NAN"
        assert ~np.any(np.isnan(y)), "FOUND NAN"
        assert ~np.any(np.isnan(h)), "FOUND NAN"
        assert ~np.any(np.isnan(bbox)), "FOUND NAN"
        bbox = np.concatenate((bbox, [[0, 0, 1, 1]]))
        class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
        input_images = np.array(image, dtype=np.float32)
        input_images = self.normalize_images(input_images)
        resid_result = self.resid_model.model.detect([input_images], verbose=0)[0]
        it_det_cent = btk_utils.resid_merge_centers(self.det_cent,
                                                    resid_result['rois'])
        if len(it_det_cent) == 0:
            it_det_cent = np.empty((0, 2))
        #iter_data = {'blend_images': data['blend_images'],
        #             'obs_condition': data['obs_condition']}
        #iter_scarlet_op = self.iter_scarlet_param.get_deblended_images(
        #    iter_data, index, peaks=it_det_cent)
        #scarlet_peaks = iter_scarlet_op['scarlet_peaks']
        #iter_deblend_image = iter_scarlet_op['scarlet_model']
        #peaks = np.unique(scarlet_peaks, axis=0)
        #return {'deblend_image': model_image, 'resid_image': resid_image,
        #        'it_det_cent': it_det_cent, 'iter_deblend_image': iter_deblend_image,
        #        'peaks': peaks, 'prediction': resid_result,
        #        'target_bbox': bbox, 'target_class': class_ids
        #        }
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': it_det_cent, 'prediction': resid_result,
                'target_bbox': bbox, 'target_class': class_ids
                }

    def normalize_images(self, images):
        images[:, :, 0:6] = (images[:, :, 0:6] - self.norm[0])/self.norm[1]
        images[:, :, 6:12] = (images[:, :, 6:12] - self.norm[2])/self.norm[3]
        return images


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
        self.scarlet_param = btk_utils.Stack_iter_params(detect_coadd=False)
        self.iter_stack = btk_utils.Stack_iter_params(detect_coadd=False)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        self.det_cent = detected_centers
        self.true_cent = np.stack([blend_list['dx'], blend_list['dy']]).T
        resid_image = blend_image - model_image
        iter_data = {'blend_images': [resid_image, ],
                     'obs_condition': [obs_cond, ]}
        stck_peaks = self.iter_stack.get_only_peaks(
            iter_data, 0)
        iter_peaks = btk_utils.stack_resid_merge_centers(self.det_cent,
                                                         stck_peaks)
        if len(iter_peaks) == 0:
            iter_peaks = np.empty((0, 2))
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': iter_peaks}


class Stack_iter_measure_params(btk.measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self, verbose=False):
        self.scarlet_param = btk_utils.Stack_iter_params(detect_coadd=True)
        self.iter_stack = btk_utils.Stack_iter_params(detect_coadd=True)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        blend_image = data['blend_images'][index]
        scarlet_op = self.scarlet_param.get_deblended_images(data, index)
        model_image = scarlet_op['scarlet_model']
        blend_list = data['blend_list'][index]
        obs_cond = data['obs_condition'][index]
        model_image[np.isnan(model_image)] = 0
        detected_centers = scarlet_op['scarlet_peaks']
        self.det_cent = detected_centers
        self.true_cent = np.stack([blend_list['dx'], blend_list['dy']]).T
        resid_image = blend_image - model_image
        iter_data = {'blend_images': [resid_image, ],
                     'obs_condition': [obs_cond, ]}
        stack_peaks = self.iter_stack.get_only_peaks(
            iter_data, 0)
        iter_peaks = btk_utils.stack_resid_merge_centers(self.det_cent,
                                                         stack_peaks)
        if len(iter_peaks) == 0:
            iter_peaks = np.empty((0, 2))
        return {'deblend_image': model_image, 'resid_image': resid_image,
                'peaks': iter_peaks}


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
        image = data['blend_images'][index],
        obs_cond = data['obs_condition'][index]
        image_array, variance_array, psf_image = btk_utils.get_stack_input(
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
        image = data['blend_images'][index],
        obs_cond = data['obs_condition'][index]
        image_array, variance_array, psf_image = btk_utils.get_stack_input(
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
