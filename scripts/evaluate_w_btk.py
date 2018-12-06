import os
import numpy as np
import sys
BTK_PATH = '/home/users/sowmyak/BlendingToolKit/'
sys.path.insert(0, BTK_PATH)
import btk

# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/resid/data'
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)
import btk_utils

# Import Mask RCNN
MRCNN_DIR = '/home/users/sowmyak/ResidualDetectron'
sys.path.append(MRCNN_DIR)  # To find local version of the library
import mrcnn.model_w_btk as modellib


def get_btk_generator():
    # Input catalog name
    catalog_name = os.path.join("/scratch/users/sowmyak/data", 'OneDegSq.fits')
    # Load parameters
    param = btk.config.Simulation_params(
        catalog_name, max_number=2, batch_size=1, seed=199)
    np.random.seed(param.seed)
    # Load input catalog
    catalog = btk.get_input_catalog.load_catlog(param)
    # Generate catlogs of blended objects
    blend_generator = btk.create_blend_generator.generate(
        param, catalog, btk_utils.resid_sampling_function)
    # Generates observing conditions for the selected survey_name & all bands
    observing_generator = btk.create_observing_generator.generate(
        param, btk_utils.resid_obs_conditions)
    # Generate images of blends in all the observing bands
    draw_blend_generator = btk.draw_blends.generate(
        param, blend_generator, observing_generator)
    meas_params = btk_utils.Scarlet_resid_params()
    meas_generator = btk.measure.generate(
        meas_params, draw_blend_generator, param)
    return meas_generator


def main(Args):
    """Test peformance for btk input blends"""
    meas_generator = get_btk_generator()
    file_name = "train" + Args.model_name
    results = []
    train = __import__(file_name)

    class InferenceConfig(train.InputConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    dataset_val = btk_utils.ResidDataset(meas_generator)
    dataset_val.load_data(training=False)
    dataset_val.prepare()
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    print("Loading weights from ", Args.model_path)
    model.load_weights(Args.model_path, by_name=True)
    for im_id in range(15):
        image1, image_meta1, gt_class_id1, gt_bbox1 =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   (im_id), use_mini_mask=False)
        true_cent = dataset_val.true_cent
        det_cent = dataset_val.det_cent
        results1 = model.detect([image1], verbose=0)
        r1 = results1[0]
        detected_centers = btk_utils.resid_merge_centers(det_cent, r1['rois'])
        det, undet, spur = btk.compute_metrics.evaluate_detection(
            detected_centers, true_cent)
        results.append([len(true_cent), det, undet, spur])
    arr_results = np.array(results).T
    print("Results: ", np.sum(arr_results, axis=1))
    save_file_name = f"detection_results_{Args.model_name}.txt"
    np.savetxt(save_file_name, arr_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', type=str,
                        help="Saved weights of model")
    args = parser.parse_args()
    main(args)
