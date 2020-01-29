# Script to train data
import os
import btk_utils
import dill
import numpy as np

# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs_oct'
# path to catlog
DATA_PATH = '/scratch/users/sowmyak/data/groups/min_snr_1/analysis/'


def main(Args):
    """Train group blends with btk"""
    norm = [0., 1, 0, 1]
    input_pull = True
    input_model_mapping = True
    max_number = 10
    count = 40000
    train_catalog_name = os.path.join(DATA_PATH,
                                      'train_dc2_i_30min_snr_1.fits')
    val_catalog_name = os.path.join(DATA_PATH, 'val_dc2_i_30min_snr_1.fits')
    train_wld_catalog_name = os.path.join(DATA_PATH,
                                          'train_group_i_30min_snr_1.fits')
    val_wld_catalog_name = os.path.join(DATA_PATH,
                                        'val_group_i_30min_snr_1.fits')
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        images_per_gpu=4, validation_for_training=True, i_mag_lim=27)
    # Load parameters for dataset and load model
    resid_model.config.WEIGHT_DECAY = 0.001
    resid_model.config.STEPS_PER_EPOCH = 1000
    resid_model.config.VALIDATION_STEPS = 20
    sampling_function = None
    obs_function = None
    selection_function = None
    layers = 'all'
    multiprocess = False
    if Args.model_name == 'model1_dc2':
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.group_sampling_function
        obs_function = btk_utils.custom_obs_condition
        selection_function = None  # btk_utils.custom_selection_function
        layers = 'all'
        norm = [0., 1., 0., 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
        max_number = 10
        resid_model.config.VALIDATION_STEPS = 10
    elif Args.model_name == 'model2_dc2':
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.group_sampling_function_numbered
        obs_function = btk_utils.custom_obs_condition
        selection_function = None  # btk_utils.custom_selection_function
        layers = 'all'
        norm = [0., 1., 0., 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
        max_number = 10
        resid_model.config.VALIDATION_STEPS = 10
        multiprocess = True
    else:
        raise AttributeError("model not found", Args.model_name)
    print("Train in model:", Args.model_name)
    resid_model.config.display()
    resid_model.make_resid_model(train_catalog_name, count=count,
                                 max_number=max_number, augmentation=True,
                                 norm_val=norm, input_pull=input_pull,
                                 sampling_function=sampling_function,
                                 obs_condition=obs_function,
                                 input_model_mapping=input_model_mapping,
                                 selection_function=selection_function,
                                 wld_catalog_name=train_wld_catalog_name,
                                 val_wld_catalog_name=val_wld_catalog_name,
                                 val_catalog_name=val_catalog_name,
                                 multiprocess=multiprocess)
    learning_rate = resid_model.config.LEARNING_RATE/10.
    np.random.seed(Args.epochs)
    history = resid_model.model.train(resid_model.dataset,
                                      resid_model.dataset_val,
                                      learning_rate=learning_rate,
                                      epochs=Args.epochs,
                                      layers=layers)
    name = Args.model_name + '_run1'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)
    learning_rate = resid_model.config.LEARNING_RATE/15.
    np.random.seed(Args.epochs + 10)
    history = resid_model.model.train(resid_model.dataset,
                                      resid_model.dataset_val,
                                      learning_rate=learning_rate,
                                      epochs=Args.epochs+10,
                                      layers=layers)
    name = Args.model_name + '_run2'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='model1', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', default=None,
                        help="Saved weights of model")
    parser.add_argument('--epochs', default=10, type=int,
                        help="Epochs to run model")
    args = parser.parse_args()
    main(args)
