# Script to train data
import os
import btk_utils
import dill
import numpy as np

# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs_oct'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'


def main(Args):
    """Train group blends with btk"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        images_per_gpu=4, validation_for_training=True)
    # Load parameters for dataset and load model
    resid_model.config.WEIGHT_DECAY = 0.001
    resid_model.config.STEPS_PER_EPOCH = 1000
    resid_model.config.VALIDATION_STEPS = 20
    if Args.model_name == 'model1':
        resid_model.config.BACKBONE = 'resnet41'
    elif Args.model_name == 'model2':
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    elif Args.model_name == 'model3':
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    print("Train in model:", Args.model_name)
    resid_model.config.display()
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=2, augmentation=True,
                                 norm_val=norm)
    learning_rate = resid_model.config.LEARNING_RATE/10.
    np.random.seed(Args.epochs)
    history = resid_model.model.train(resid_model.dataset,
                                      resid_model.dataset_val,
                                      learning_rate=learning_rate,
                                      epochs=Args.epochs,
                                      layers='all')
    name = Args.model_name + '_run7'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)
    #learning_rate = resid_model.config.LEARNING_RATE/5.
    #np.random.seed(Args.epochs + 20)
    #history = resid_model.model.train(resid_model.dataset,
    #                                  resid_model.dataset_val,
    #                                 learning_rate=learning_rate,
    #                                 epochs=Args.epochs+20,
    #                                  layers='all')
    #name = Args.model_name + '_run7'
    #with open(name + ".dill", 'wb') as handle:
    #    dill.dump(history.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='model1', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', default=None, type=str,
                        help="Saved weights of model")
    parser.add_argument('--epochs', default=40, type=int,
                        help="Epochs to run model")
    args = parser.parse_args()
    main(args)
