# Script to train data
#  #use utils.generate_anchors_no_spill
import os
import btk_utils
import dill
import astropy.table


# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'


def get_wld_catalog():
    """Returns pre-run wld catalog for group identification"""
    wld_catalog_name = os.path.join(
        DATA_PATH, 'wld_1sqdeg_lsst_i_catalog.fits')
    wld_catalog = astropy.table.Table.read(wld_catalog_name, format='fits')
    selected_gal = wld_catalog[
        (wld_catalog['sigma_m'] < 2) & (wld_catalog['ab_mag'] < 28)]
    return selected_gal


def main(Args):
    """Train 2 gal blends with btk"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 2000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    new_model_name = "resid_" + Args.model_name + "_btk_group"
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        new_model_name=new_model_name, images_per_gpu=4)
    # Load parameters for dataset and load model
    resid_model.make_resid_model(
        catalog_name, count=count,
        max_number=6, augmentation=True,
        sampling_functon=btk_utils.group_sampling_function,
        selection_function=btk_utils.basic_selection_function,
        wld_catalog=get_wld_catalog(),
        norm_val=norm)
    learning_rate = resid_model.config.LEARNING_RATE*5
    history1 = resid_model.model.train(resid_model.dataset,
                                       resid_model.dataset_val,
                                       learning_rate=learning_rate,
                                       epochs=10,
                                       layers='all')
    name = new_model_name + '_run1_loss'
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
