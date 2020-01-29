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
    input_pull = False
    input_model_mapping = False
    max_number = 2
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
    sampling_function = None
    layers = 'all'
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
    elif Args.model_name == 'model4':
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    elif Args.model_name == 'model5':
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet35'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    elif Args.model_name == 'model4_large':
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = '4+'  # '3+'
    elif Args.model_name == 'model6':
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0, 1, 51.2789974336363, 1038.4760551905683]
        input_pull = True
    elif Args.model_name == 'model7':
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model8':    # stretch = 0.1, Q = 3
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model9':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model10':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1., 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model10_again':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model10_again2':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = None
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model10_again3':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model10_2':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0., 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model11':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1., 0., 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model11_2':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1., 0., 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
    elif Args.model_name == 'model12':    # stretch = 2000, Q = 0.5
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
        max_number = 6
    elif Args.model_name == 'model12_again':    # stretch = 2000, Q = 0.5 # larger learning rate
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
        max_number = 10 # changed from 6 to 10 for run 4
    elif Args.model_name == 'model12_again2':    # stretch = 2000, Q = 0.5 # larger learning rate val set reduced to 10
        resid_model.config.TRAIN_BN = False
        resid_model.config.BACKBONE = 'resnet41'
        resid_model.config.SKIP_P2_RPN = True
        resid_model.config.BACKBONE_STRIDES = [8, 16, 32, 64]
        resid_model.config.RPN_ANCHOR_SCALES = (8, 16, 32, 64)
        sampling_function = btk_utils.resid_general_sampling_function_large
        layers = 'all'
        norm = [0., 1.45, 0, 1.]  # [0, 1, 0, 1]
        input_pull = True
        input_model_mapping = True
        max_number = 6
        resid_model.config.VALIDATION_STEPS = 10
    else:
        raise AttributeError("model not found", Args.model_name)
    print("Train in model:", Args.model_name)
    resid_model.config.display()
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=max_number, augmentation=True,
                                 norm_val=norm, input_pull=input_pull,
                                 sampling_function=sampling_function,
                                 input_model_mapping=input_model_mapping)
    learning_rate = resid_model.config.LEARNING_RATE/10.
    np.random.seed(Args.epochs)
    history = resid_model.model.train(resid_model.dataset,
                                      resid_model.dataset_val,
                                      learning_rate=learning_rate,
                                      epochs=Args.epochs,
                                      layers=layers)
    name = Args.model_name + '_run2'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)
    learning_rate = resid_model.config.LEARNING_RATE/10.
    np.random.seed(Args.epochs + 10)
    history = resid_model.model.train(resid_model.dataset,
                                      resid_model.dataset_val,
                                      learning_rate=learning_rate,
                                      epochs=Args.epochs+10,
                                      layers=layers)
    name = Args.model_name + '_run3'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='model1', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', default=None,
                        help="Saved weights of model")
    parser.add_argument('--epochs', default=40, type=int,
                        help="Epochs to run model")
    args = parser.parse_args()
    main(args)
