import os
import numpy as np
import sys
BTK_PATH = '/home/users/sowmyak/BlendingToolKit/'
sys.path.insert(0, BTK_PATH)
import btk

# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)
import btk_utils


def detection_i_band(Args):
    """Test performance for btk input blends"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 15#4000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=False,
        images_per_gpu=1)
    # Load parametrs for dataset and load model
    meas_params = btk_utils.Scarlet_resid_params(detect_coadd=False)
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=2, norm_val=norm,
                                 meas_params=meas_params)
    results = []
    # np.random.seed(0)
    for im_id in range(count):
        iter_detected, sep_detected, true = resid_model.get_detections(im_id)
        for i in range(len(true)):
            it_det, it_undet, it_spur = btk.compute_metrics.evaluate_detection(
                iter_detected[i], true[i])
            # print(it_det, it_undet, it_spur)
            if len(sep_detected[i]) == 0:
                sep_det, sep_undet, sep_spur = 0, len(true[i]), 0
            else:
                unique_sep_det_cent = np.unique(sep_detected[i], axis=0)
                sep_det, sep_undet, sep_spur = btk.compute_metrics.evaluate_detection(
                        unique_sep_det_cent, true[i])
            # print(sep_det, sep_undet, sep_spur)
            results.append(
                [len(true[i]), it_det, it_undet, it_spur, sep_det, sep_undet, sep_spur])
    arr_results = np.array(results).T
    print("Results: ", np.sum(arr_results, axis=1))
    save_file_name = f"sep_det_results_2gal_i_band_temp.txt"
    np.savetxt(save_file_name, arr_results)


def detection_coadd(Args):
    """Test performance for btk input blends"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 15#4000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=False,
        images_per_gpu=1)
    # Load parametrs for dataset and load model
    meas_params = btk_utils.Scarlet_resid_params(detect_coadd=True)
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=2, norm_val=norm,
                                 meas_params=meas_params)
    results = []
    # np.random.seed(0)
    for im_id in range(count):
        iter_detected, sep_detected, true = resid_model.get_detections(im_id)
        for i in range(len(true)):
            it_det, it_undet, it_spur = btk.compute_metrics.evaluate_detection(
                iter_detected[i], true[i])
            # print(it_det, it_undet, it_spur)
            if len(sep_detected[i]) == 0:
                sep_det, sep_undet, sep_spur = 0, len(true[i]), 0
            else:
                unique_sep_det_cent = np.unique(sep_detected[i], axis=0)
                sep_det, sep_undet, sep_spur = btk.compute_metrics.evaluate_detection(
                        unique_sep_det_cent, true[i])
            # print(sep_det, sep_undet, sep_spur)
            results.append(
                [len(true[i]), it_det, it_undet, it_spur, sep_det, sep_undet, sep_spur])
    arr_results = np.array(results).T
    print("Results: ", np.sum(arr_results, axis=1))
    save_file_name = f"sep_det_results_2gal_coadd_temp.txt"
    np.savetxt(save_file_name, arr_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', type=str,
                        help="Saved weights of model")
    args = parser.parse_args()
    detection_coadd(args)
    detection_i_band(args)
