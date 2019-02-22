# Script to train data
##use utils.generate_anchors_no_spill
import os
import btk_utils
import dill
import numpy as np

# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'


def main(Args):
    """Train group blends with btk"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 2000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    new_model_name = "resid_btk_square"
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        new_model_name=new_model_name, images_per_gpu=4,
        validation_for_training=True)
    # Load parametrs for dataset and load model
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=2, augmentation=True,
                                 norm_val=norm)
    learning_rate = resid_model.config.LEARNING_RATE/20.
    #resid_model.config.LEARNING_MOMENTUM = 0.92
    resid_model.config.WEIGHT_DECAY = 0.001
    resid_model.config.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])/2.
    resid_model.config.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])/2.
    #resid_model.config.display()
    history1 = resid_model.model.train(resid_model.dataset,
                                       resid_model.dataset_val,
                                       learning_rate=learning_rate,
                                       epochs=150,
                                       layers='all')
    name = new_model_name + '_run5_12_loss'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history1.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='15_8', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', default=None, type=str,
                        help="Saved weights of model")
    args = parser.parse_args()
    main(args)
