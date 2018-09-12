# Script to train data

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import dill
import pandas as pd
import tensorflow as tf


# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/lavender/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/lavender/resid_detection'
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)

MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/resid120180911T1542/mask_rcnn_resid1_0020.h5'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
# from mrcnn.model import log


class InputConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "resid1"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8  # 32

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
    STEPS_PER_EPOCH = 500  # 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 40  # 10
    # modifications here
    LEARNING_RATE = 0.0001
    DETECTION_MIN_CONFIDENCE = 0.9
    # MEAN_PIXEL = np.array([0., 0., 0.])
    USE_MINI_MASK = False
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.3, 0.3])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.3, 0.3])
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 200
    # Modification!
    MEAN_PIXEL = np.zeros(12)
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # RPN_ANCHOR_RATIOS = [1, ]


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
        if training:
            filename = os.path.join(DATA_PATH, "resid_images_norm_train.h5")
            with h5py.File(filename, 'r') as hf:
                self.X = hf['resid_images'][:]
            print(self.X.shape)
            assert not np.any(np.isnan(self.X))
            filename = os.path.join(DATA_PATH, "resid_cat_train.csv")
            self.Y = pd.read_csv(filename)
        else:
            filename = os.path.join(DATA_PATH, "resid_images_norm_val.h5")
            with h5py.File(filename, 'r') as hf:
                self.X = hf['resid_images'][:]
            print(self.X.shape)
            assert not np.any(np.isnan(self.X))
            filename = os.path.join(DATA_PATH, "resid_cat_val.csv")
            self.Y = pd.read_csv(filename)
        if count is None:
            count = len(self.X)
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
        image = self.X[image_id, :, :, :]
        return image  # rgb_image

    def load_bbox(self, image_id):
        """Generate bbox of undetcted object
        """
        x0 = self.Y.at[image_id, 'detect_x']
        y0 = self.Y.at[image_id, 'detect_y']
        h = self.Y.at[image_id, 'detect_h']
        bbox = [[y0 - h / 2., x0 - h / 2., y0 + h / 2., x0 + h / 2.]]
        class_id = np.array([1, ], dtype=np.int32)
        return np.array(bbox, dtype=np.int32), class_id

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "resid":
            return info["object"]
        else:
            super(self.__class__).image_reference(self, image_id)


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


def main():
    tf.set_random_seed(0)
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

    history2 = model.train(dataset_train, dataset_val,
                           learning_rate=config.LEARNING_RATE,
                           epochs=40,
                           layers='all')
    name = config.NAME + '_run2_loss'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history2.history, handle)


if __name__ == "__main__":
    main()
