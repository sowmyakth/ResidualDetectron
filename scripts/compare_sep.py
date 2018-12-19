import sep
import sys
import os
import numpy as np
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


class Sep_params(btk.measure.Measurement_params):
    def get_centers(self, image):
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        return np.stack((catalog['x'], catalog['y']), axis=1)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        peaks = self.get_centers(images)
        return [None, peaks]


def get_btk_generator():
    # Input catalog name
    catalog_name = os.path.join("/scratch/users/sowmyak/data", 'OneDegSq.fits')
    # Load parameters
    param = btk.config.Simulation_params(
        catalog_name, max_number=2, batch_size=1, seed=199)
    np.random.seed(param.seed)
    # Load input catalog
    catalog = btk.get_input_catalog.load_catlog(param)
    # Generate catalogs of blended objects
    blend_generator = btk.create_blend_generator.generate(
        param, catalog, btk_utils.resid_general_sampling_function)
    # Generates observing conditions for the selected survey_name & all bands
    observing_generator = btk.create_observing_generator.generate(
        param, btk_utils.resid_obs_conditions)
    # Generate images of blends in all the observing bands
    draw_blend_generator = btk.draw_blends.generate(
        param, blend_generator, observing_generator)
    meas_params = Sep_params()
    meas_generator = btk.measure.generate(
        meas_params, draw_blend_generator, param)
    return meas_generator


def run_sep():
    """Test performance for btk input blends"""
    meas_generator = get_btk_generator()
    results = []
    np.random.seed(0)
    for im_id in range(4000):
        output, deb, _ = next(meas_generator)
        blend_list = output['blend_list'][0]
        detected_centers = deb[0][1]
        true_centers = np.stack([blend_list['dx'], blend_list['dy']]).T
        det, undet, spur = btk.compute_metrics.evaluate_detection(
            detected_centers, true_centers)
        print(len(true_centers), det, undet, spur)
        results.append([len(true_centers), det, undet, spur])
    arr_results = np.array(results).T
    print("Results: ", np.sum(arr_results, axis=1))
    save_file_name = f"detection_results_sep.txt"
    np.savetxt(save_file_name, arr_results)


if __name__ == '__main__':
    run_sep()
