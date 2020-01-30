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
import mrcnn
import mrcnn.config_btk_only
from btk_utils import custom_obs_condition, group_sampling_function_numbered


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

class ResidDataset(mrcnn.utils.Dataset):
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
            x, y, h = btk_utils.get_undetected(blend_list, detected_centers,
                                               self.obs_cond[i][3],
                                               self.i_mag_lim)
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

    def make_resid_model(self, catalog_name, count=256,
                         sampling_function=None, max_number=2,
                         augmentation=False, norm_val=None,
                         selection_function=None, wld_catalog_name=None,
                         meas_params=None, input_pull=False,
                         input_model_mapping=False, obs_condition=None,
                         val_wld_catalog_name=None, val_catalog_name=None,
                         multiprocess=False):
        """Creates dataset and loads model"""
        # If no user input sampling function then set default function
        import mrcnn.model_btk_only as model_btk
        if not sampling_function:
            sampling_function = btk_utils.resid_general_sampling_function
        if wld_catalog_name:
            train_wld_catalog = astropy.table.Table.read(
                wld_catalog_name, format='fits')
        else:
            train_wld_catalog = None
        self.meas_generator = btk_utils.make_meas_generator(
            catalog_name, self.config.BATCH_SIZE, max_number,
            sampling_function, selection_function, train_wld_catalog,
            meas_params, obs_condition, multiprocess)
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
                if val_wld_catalog_name:
                    val_wld_catalog = astropy.table.Table.read(
                        val_wld_catalog_name, format='fits')
                else:
                    val_wld_catalog = None
                val_meas_generator = btk_utils.make_meas_generator(
                    val_catalog_name, self.config.BATCH_SIZE, max_number,
                    sampling_function, selection_function, val_wld_catalog,
                    meas_params, obs_condition, multiprocess)
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

class RD_metric_params(btk.compute_metrics.Metrics_params):
    def __init__(self, *args, **kwargs):
        super(RD_metric_params, self).__init__(*args, **kwargs)
        """Class describing functions to return results of
        detection/deblending/measurement algorithm in meas_generator. Each
        time the algorithm is called, it is run on a batch of blends yielded
        by the meas_generator.
    """

    def get_detections(self):
        """Returns input blend catalog and detection catalog for
        the detection performed.

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
        blend_op, deblend_op, _ = next(self.meas_generator)
        true_tables = blend_op['blend_list']
        detected_tables = []
        for i in range(len(true_tables)):
            detected_centers = deblend_op[i]['peaks']
            detected_table = astropy.table.Table(detected_centers,
                                                 names=['dx', 'dy'])
            detected_table['scarlet_multi_fit'] = deblend_op[i]['scarlet_mf']
            detected_tables.append(detected_table)
        return true_tables, detected_tables
