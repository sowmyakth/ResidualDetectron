# Script to train data

import os
import sys
import numpy as np
import h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dill

# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'

# Directory to save logs and trained model

MODEL_DIR = '/scratch/users/sowmyak/lavender/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/lavender'

CODE_PATH = '/home/users/sowmyak/NN_blend/scripts'
sys.path.append(CODE_PATH)




MODEL_PATH = None
# MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_final20180608T2004/mask_rcnn_blend_final_0050.h5'
#MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_final_again20180608T2004/mask_rcnn_blend_final_again_0122.h5'
#MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_test20180717T1318/mask_rcnn_blend_test_0045.h5'
#MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_test_again20180719T1635/mask_rcnn_blend_test_again_0058.h5'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn.model import log


def normalize(im1, im2, c_index=2):
    vmin = im1.min(axis=0).min(axis=0)[c_index]
    vmax = im1.max(axis=0).max(axis=0)[c_index]
    vmean = im1.mean(axis=0).mean(axis=0)
    result = np.zeros_like(im2)
    result[:] = im2
    result[result < vmin] = vmin
    result[result > vmax] = vmax
    result = (result - vmin) / (vmax - vmin)
    return result


def normalize2(im, vmin, vmax, c_index=None):
    result = np.zeros_like(im)
    result[:] = im
    if c_index is None:
        for i in range(im.shape[2]):
            result[:,:,i][result[:,:,i] < vmin] = vmin
            result[:,:,i][result[:,:,i] > vmax] = vmax
            result[:,:,i] = (result[:,:,i] - vmin) / (vmax - vmin)
    else: 
        result[result < vmin] = vmin
        result[result > vmax] = vmax
        result = (result - vmin) / (vmax - vmin)
    return result * 255 # np.array(result * 255, dtype=np.uint8)


class InputConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "blend_new_loss_relu"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8 #32

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 16

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500  #200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 40  #10
    # modifications here
    LEARNING_RATE = 0.001
    DETECTION_MIN_CONFIDENCE = 0.8
    # MEAN_PIXEL = np.array([0., 0., 0.])
    USE_MINI_MASK = False
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
    }
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.3, 0.3])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.3, 0.3])
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 500


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_data(self, training=True, count=None):
        """loads traininf and test input and output data
        Keyword Arguments:
            filename -- numpy file where data is saved
        """
        # filename = os.path.join(DATA_PATH, 'lavender_temp/stamps2.pickle')
        # filename = os.path.join(DATA_PATH, 'lavender_temp/stamps2.dill')
        # with open(filename, 'rb') as handle:
        #    data = pickle.load(handle)
        #    data = dill.load(handle)
        self.X, self.Y = {}, {}
        x_names = ('blend_image', 'loc2', 'loc1')
        y_names = ('Y1', 'Y2')
        if training:
            for name in x_names:
                filename = os.path.join(DATA_PATH,
                                        'lavender_temp/train_' + name + '.h5')
                with h5py.File(filename, 'r') as hf:
                    self.X[name] = hf[name][:]
                print(self.X[name].shape)
                assert not np.any(np.isnan(self.X[name]))
            for name in y_names:
                filename = os.path.join(DATA_PATH,
                                        'lavender_temp/train_' + name + '.h5')
                with h5py.File(filename, 'r') as hf:
                    self.Y[name] = hf[name][:]
                assert not np.any(np.isnan(self.Y[name]))
        else:
            for name in x_names:
                filename = os.path.join(DATA_PATH,
                                        'lavender_temp/val_' + name + '.npy')
                self.X[name] = np.load(filename)
                assert not np.any(np.isnan(self.X[name]))
            for name in y_names:
                filename = os.path.join(DATA_PATH,
                                        'lavender_temp/val_' + name + '.npy')
                self.Y[name] = np.load(filename)
                assert not np.any(np.isnan(self.Y[name]))
        if count is None:
            count = len(self.X['blend_image'])
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

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        returns RGB Image [height, width, bands]
        """
        image = self.X['blend_image'][image_id, :, :, :]
        return image  # rgb_image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "blend":
            return info["object"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        mask1 = self.X['loc1'][image_id]
        mask2 = self.X['loc2'][image_id]
        mask = np.dstack([mask1, mask2])
        # Map class names to class IDs.
        class_ids = np.array([1, 1])
        return mask.astype(np.bool), class_ids.astype(np.int32)


def plot_history(history, string='training'):
    print(history.history.keys())
    loss_names = ['rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss',
                  'mrcnn_bbox_loss']
    fig, ax = plt.subplots(1, 3, figsize=(14, 8))
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_title('model loss')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(loc='upper left')
    for name in loss_names:
        ax[1].plot(history.history[name], label='train_' + name)
    ax[1].set_title('train loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(loc='upper left')
    for name in loss_names:
        ax[2].plot(history.history['val_' + name], label='val_' + name)
    ax[2].set_title('val loss')
    ax[2].set_ylabel('loss')
    ax[2].set_xlabel('epoch')
    ax[2].legend(loc='upper left')
    name = string + '_loss'
    fig.savefig(name)
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)


def main():
    config = InputConfig()
    config.display()
    # Training dataset# Train
    dataset_train = ShapesDataset()
    dataset_train.load_data()
    dataset_train.prepare()
    # import ipdb;ipdb.set_trace()

    # Validation dataset
    dataset_val = ShapesDataset()
    dataset_val.load_data(training=False)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?# Which
    if MODEL_PATH:
        model_path = MODEL_PATH
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
    # Training - Stage 1
    print("head layers")
    history1 = model.train(dataset_train, dataset_val,
                           learning_rate=config.LEARNING_RATE,
                           epochs=40,
                           layers='all')
    plot_history(history1, config.NAME + '_run1')



if __name__ == "__main__":
    main()
