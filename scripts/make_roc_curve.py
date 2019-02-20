import os
import numpy as np
from scipy import spatial
import btk_utils
import dill
import pandas as pd
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'


def get_undetected(true_cat, meas_cent, obs_cond, distance_upper_bound=10):
    """Returns bounding boxes for galaxies that were undetected. The bounding
    boxes are square with height set as twice the PSF convolved HLR. Since
    CatSim catalog has separate bulge and disk parameters, the galaxy HLR is
    approximated as the flux weighted average of bulge and disk HLR.

    The function returns the x and y coordinates of the lower left corner of
    box, and the height of each undetected galaxy. A galaxy is marked as
    undetected if no detected center lies within distance_upper_bound of it's
    true center.
    Args:
        true_cat: CatSim-like catalog of true galaxies.
        meas_cent: ndarray of x and y coordinates of detected centers.
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        distance_upper_bound: Distance up-to which a detected center can be
                              matched to a true center.
    Returns:
        Bounding box of undetected galaxies

    """
    psf_sigma = obs_cond.zenith_psf_fwhm*obs_cond.airmass**0.6
    pixel_scale = obs_cond.pixel_scale
    peaks = np.stack([true_cat['dx'], true_cat['dy']]).T
    if len(peaks) == 0:
        undetected = range(len(true_cat))
    else:
        z_tree = spatial.KDTree(peaks)
        meas_cent = np.array(meas_cent).reshape(-1, 2)
        match = z_tree.query(meas_cent,
                             distance_upper_bound=distance_upper_bound)
        undetected = np.setdiff1d(range(len(true_cat)), match[1])
    numer = true_cat['a_d']*true_cat['b_d']*true_cat['fluxnorm_disk'] + true_cat['a_b']*true_cat['b_b']*true_cat['fluxnorm_bulge']
    hlr = numer / (true_cat['fluxnorm_disk'] + true_cat['fluxnorm_bulge'])
    assert ~np.any(np.isnan(hlr)), "FOUND NAN"
    h = np.hypot(hlr, 1.18*psf_sigma)*2 / pixel_scale
    h = np.array(h, dtype=np.int32)
    x0 = true_cat['dx'] - h/2
    y0 = true_cat['dy'] - h/2
    return x0[undetected], y0[undetected], h[undetected], undetected


def load_input(dataset):
    """Generates image + bbox for undetected objects if any"""
    output, deb, _ = next(dataset.meas_generator)
    blend_list = output['blend_list'][0]
    obs_cond = output['obs_condition'][0]
    input_images, input_bboxes, input_class_ids = [], [], []
    for i in range(len(output['blend_list'])):
        blend_images = output['blend_images'][i]

    model_images = deb[0][0]
    model_images[np.isnan(model_images)] = 0
    detected_centers = deb[0][1]
    resid_images = blend_images - model_images
    image = np.dstack([resid_images, model_images])
    x, y, h, undetect = get_undetected(blend_list, detected_centers,
                                       obs_cond[3])
    bbox = np.array([y, x, y+h, x+h], dtype=np.int32).T
    assert ~np.any(np.isnan(bbox)), "FOUND NAN"
    bbox = np.concatenate((bbox, [[0, 0, 1, 1]]))
    class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
    input_images.append(image)
    input_bboxes.append(bbox)
    input_class_ids.append(class_ids)
    input_images = np.array(input_images, dtype=np.float32)
    input_images = dataset.normalize_images(input_images)
    assert ~np.any(np.isnan(input_images)), "FOUND NAN"
    output = [input_images, input_bboxes, input_class_ids,
              blend_list[undetect], detected_centers]
    return output


def get_detections(model):
    """
    Returns model detected centers and true center for data entry index.
    Args:
        index: Index of dataset to perform detection on.
    Returns:
        x and y coordinates of iteratively detected centers, centers
        detected initially and true centers.
    Useful for evaluating model detection performance."""
    image, gt_bbox, gt_class_id, blend_list, detected_centers = load_input(
        model.dataset)
    results = model.detect(image, verbose=0)[0]
    return results, blend_list, detected_centers


def run_model_without_threshold(Args):
    """Test performance for btk input blends"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 4000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.Resid_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=False,
        images_per_gpu=1)
    resid_model.config.DETECTION_MIN_CONFIDENCE = 0
    resid_model.config.display()
    # Load parametrs for dataset and load model
    meas_params = btk_utils.Scarlet_resid_params(detect_coadd=True)
    resid_model.make_resid_model(catalog_name, count=count,
                                 max_number=2, norm_val=norm,
                                 meas_params=meas_params)
    results = {'blend_list': [], 'detected_centers': []}
    # np.random.seed(0)
    for i in range(count):
        r1, blend_list1, detected_centers1 = get_detections(resid_model)
        results[i]['model_op'] = r1
        results[i]['blend_list'] = blend_list1
        results[i]['detected_centers'] = detected_centers1
    name = f"model_without_threshold"
    with open(name + ".dill", 'wb') as handle:
        dill.dump(results, handle)


def match_detections(detected_centers, true_centers,
                     distance_upper_bound=3):
    if len(detected_centers) == 0:
        # no detection
        return np.array([]), np.array(range(len(true_centers))), np.array([])
    z_tree = spatial.KDTree(true_centers)
    detected_centers = np.array(detected_centers).reshape(-1, 2)
    match = z_tree.query(detected_centers,
                         distance_upper_bound=distance_upper_bound)
    fin, = np.where(match[0] != np.inf)  # match exists
    inf, = np.where(match[0] == np.inf)  # no match within distance_upper_bound
    detected = np.unique(match[1][fin])
    undetected = np.setdiff1d(range(len(true_centers)), match[1][fin])
    spurious = np.unique(match[1][inf])
    return detected, undetected, spurious


def get_roc_point(filename, threshold):
    columns = ['total', 'detected', 'undetected', 'spurious']
    columns += ['tp', 'fp', 'tn', 'fn']
    with open(filename, 'rb') as handle:
        results = dill.load(handle)
    output = pd.DataFrame(np.zeros((len(results), len(columns))),
                          columns=columns)
    for i in results:
        true_centers = np.stack(
            [results[i]['blend_list']['dx'], results[i]['blend_list']['dy']]).T
        q, = np.where(results[i]['model_op']['score'] > threshold)
        detected_centers = results[i]['detected_centers']
        iter_detected_centers = btk_utils.resid_merge_centers(
                detected_centers, results[i]['model_op']['rois'][q],
                center_shift=0)
        detected, undetected, spurious = match_detections(
            iter_detected_centers, true_centers)
        output.loc[i, 'total'] = len(true_centers)
        output.loc[i, 'detected'] = len(detected)
        output.loc[i, 'undetected'] = len(undetected)
        output.loc[i, 'spurious'] = len(spurious)
        output.loc[i, 'tp'] = len(detected)
        output.loc[i, 'fn'] = len(undetected)
        if (len(true_centers) == 0) & (len(spurious) == 0):
            output.loc[i, 'fp'] = 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', type=str,
                        help="Saved weights of model")
    args = parser.parse_args()
    run_model_without_threshold(args)
