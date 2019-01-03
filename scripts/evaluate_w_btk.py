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


def main(Args):
    """Test performance for btk input blends"""
    count = 4000
    catalog_name = os.path.join("/scratch/users/sowmyak/data", 'OneDegSq.fits')
    resid_model = btk_utils.Resid_metrics_model()
    resid_model.make_resid_model(Args.model_name, Args.model_path,
                                 MODEL_DIR, catalog_name, count=count,
                                 max_number=2)
    results = []
    np.random.seed(0)
    for im_id in range(count):
        detected_centers, true_centers = resid_model.get_detections(im_id)
        det, undet, spur = btk.compute_metrics.evaluate_detection(
            detected_centers, true_centers)
        print(det, undet, spur)
        results.append([len(true_centers), det, undet, spur])
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
