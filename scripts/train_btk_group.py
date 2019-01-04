# Script to train data
#  #use utils.generate_anchors_no_spill
import os
import sys
import dill


# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/resid/data'
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)
import btk_utils
# from mrcnn.model import log


def main(Args):
    """Test performance for btk input blends"""
    count = 40000
    catalog_name = os.path.join("/scratch/users/sowmyak/data", 'OneDegSq.fits')
    resid_model = btk_utils.Resid_metrics_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        new_model_name=Args.model_name + "_btk", images_per_gpu=8)
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=10)
    learning_rate = resid_model.config.LEARNING_RATE
    history1 = resid_model.model.train(resid_model.dataset, None,
                                       learning_rate=learning_rate,
                                       epochs=20,
                                       layers='all')
    name = resid_model.config.NAME + '_run1_loss'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history1.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', type=str,
                        help="Saved weights of model")
    args = parser.parse_args()
    main(args)
