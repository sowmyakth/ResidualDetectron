import os
import sys
import numpy as np
import pandas as pd
# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/resid/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/resid/results'
CODE_PATH = '/home/users/sowmyak/ResidualDetectron/scripts'
sys.path.append(CODE_PATH)
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


# Change this for differnt models
import train6 as train
from train6 import InputConfig
import mrcnn.model as modellib

#MODEL_PATH ='/scratch/users/sowmyak/resid/logs/resid1_again20181008T1132/mask_rcnn_resid1_again_0060.h5'
#MODEL_PATH = '/scratch/users/sowmyak/resid/logs/resid320180926T1459/mask_rcnn_resid3_0060.h5'
MODEL_PATH = '/scratch/users/sowmyak/resid/logs/resid620181008T1416/mask_rcnn_resid6_0060.h5'


def get_val_set(dataset_val):
    dataset_val.Y['other_x0'] = pd.Series(np.zeros(len(dataset_val.Y)),
                                          index=dataset_val.Y.index)
    dataset_val.Y['other_y0'] = pd.Series(np.zeros(len(dataset_val.Y)),
                                          index=dataset_val.Y.index)
    dataset_val.Y['other_h'] = pd.Series(np.zeros(len(dataset_val.Y)),
                                         index=dataset_val.Y.index)
    q1, = np.where(dataset_val.Y['input_indxs'] == 0)
    dataset_val.Y.loc[q1, ['other_x0']] = dataset_val.Y['gal1_x_tru'][q1]
    dataset_val.Y.loc[q1, ['other_y0']] = dataset_val.Y['gal1_y_tru'][q1]
    dataset_val.Y.loc[q1, ['other_h']] = np.hypot(dataset_val.Y['gal1_sigma_tru'][q1] * 2.35, 0.67) / 0.2
    q2, = np.where(dataset_val.Y['input_indxs'] == 1)
    dataset_val.Y.loc[q2, ['other_x0']] = dataset_val.Y['gal2_x_tru'][q2]
    dataset_val.Y.loc[q2, ['other_y0']] = dataset_val.Y['gal2_y_tru'][q2]
    dataset_val.Y.loc[q2, ['other_h']] = np.hypot(dataset_val.Y['gal1_sigma_tru'][q2] * 2.35, 0.67) / 0.2
    ids, = np.where((dataset_val.Y['input_indxs'] != 2))
    dataset_val.Y = dataset_val.Y.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    print(f"select {len(ids)} images for analysis from {len(dataset_val.Y)} images in val set ")
    return dataset_val, ids


def make_new_table(num):
    """Makes astropy table of default values with rows for each
    network prediction"""
    columns = ('id', 'network_score', 'iou',
               'ds_undet', 'ds_other',
               'unrecog_blnd', 'iter_recog', 'iter_recog_other',
               'iter_recog_both',
               'iter_spurious', 'iter_shred', 'check',
               'metric_score', 'number_pred',
               'roi_y1', 'roi_x1', 'roi_y2', 'roi_x2',)
    d = {}
    for c in columns:
        d[c] = np.zeros(num)
    tab = pd.DataFrame(d)
    return tab


class InferenceConfig(InputConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_unit_distances(cat, result, image_id):
    pred_x0 = (result['rois'][:, 3] + result['rois'][:, 1]) / 2.
    pred_y0 = (result['rois'][:, 2] + result['rois'][:, 0]) / 2.
    gt_x01 = cat['detect_x'][image_id] + cat['detect_h'][image_id] / 2.
    gt_y01 = cat['detect_y'][image_id] + cat['detect_h'][image_id] / 2.
    gt_x02, gt_y02 = cat['other_x0'][image_id], cat['other_y0'][image_id]
    # scrlt_x0, scrlt_y0 = cat['scrlt_y0'][image_id], cat['scrlt_x0'][image_id]
    # dist1 = np.hypot((scrlt_x0 - gt_x01), (scrlt_y0 - gt_y01))
    # dist2 = np.hypot((scrlt_x0 - gt_x02), (scrlt_y0 - gt_y02))
    # ds1 = np.hypot((pred_x0 - gt_x01), (pred_y0 - gt_y01)) / dist1
    # ds2 = np.hypot((pred_x0 - gt_x02), (pred_y0 - gt_y02)) / dist2
    ds1 = np.hypot((pred_x0 - gt_x01), (pred_y0 - gt_y01)) / cat['detect_h'][image_id]
    ds2 = np.hypot((pred_x0 - gt_x02), (pred_y0 - gt_y02)) / cat['other_h'][image_id]
    return ds1, ds2


def get_stats(ids, model, inference_config, dataset_val):
    main_tab = make_new_table(0)
    for image_id in ids:
        clss = 'check'
        original_image, image_meta, gt_class_id, gt_bbox =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        result = model.detect([original_image])[0]
        num = len(result['scores'])
        if num == 0:
            tab = make_new_table(1)
            tab.at[:, ['id']] = image_id
            tab.at[:, ['unrecog_blnd']] = 1
            main_tab = pd.concat([main_tab, tab], ignore_index=True)
            continue
        tab = make_new_table(num)
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_bbox, gt_class_id, result['rois'], result['class_ids'],
            result['scores'],
            iou_threshold=0., score_threshold=0.0)
        ds1, ds2 = get_unit_distances(dataset_val.Y, result, image_id)
        # Check to see if both det1 and det2 are not set by same prediction!!!!
        det1 = np.argmin(ds1) if np.min(ds1) < 0.2 else -1
        det2 = np.argmin(ds2) if np.min(ds2) < 0.2 else -1
        det2 = -1 if det1 == det2 else det2
        if (num == 1) & (det1 != -1):
            clss = 'iter_recog'
            tab.at[:, ['metric_score']] = 1
        elif (num == 1) & (det2 != -1):
            clss = 'iter_recog_other'
            tab.at[:, ['metric_score']] = 1
        elif ((num == 2) & (det1 != -1) & (det2 != -1)):
            clss = 'iter_recog_both'
            tab.at[:, ['metric_score']] = 0.5
        else:
            if np.any(overlaps[0] == 0) or (num == 1):
                clss = 'iter_spurious'
            else:
                clss = 'iter_shred'
            tab.at[:, ['metric_score']] = -1
            if ((det1 != -1) & (det2 != -1)):
                tab.at[det1, ['metric_score']] = 0.5
                tab.at[det2, ['metric_score']] = 0.5
            elif det1 != -1:
                tab.at[det1, ['metric_score']] = 1
            elif det2 != -1:
                tab.at[det2, ['metric_score']] = 1
        for j in range(num):
            tab.at[j, ['id']] = image_id
            tab.at[j, [clss]] = 1
            tab.at[j, ['network_score']] = result['scores'][j]
            tab.at[j, ['iou']] = overlaps[j][0]
            tab.at[j, ['number_pred']] = num
            tab.at[j, ['ds_undet']] = ds1[j]
            tab.at[j, ['ds_other']] = ds2[j]
            tab.at[j, ['roi_y1']] = result['rois'][j][0]
            tab.at[j, ['roi_x1']] = result['rois'][j][1]
            tab.at[j, ['roi_y2']] = result['rois'][j][2]
            tab.at[j, ['roi_x2']] = result['rois'][j][3]
        main_tab = pd.concat([main_tab, tab], ignore_index=True)
    return main_tab


def main():
    # Load validation set data
    dataset_val = train.ShapesDataset()
    dataset_val.load_data(training=False)
    dataset_val.prepare()
    inference_config = InferenceConfig()
    val_set, ids = get_val_set(dataset_val)
    num = len(ids)
    print(f"Getting results for {num} residuals in validation set")
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    print("Loading weights from ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)
    summ_results = get_stats(ids, model,
                             inference_config, val_set)
    out_name = inference_config.NAME + '_metric.pd'
    pd_file_name = os.path.join(DATA_PATH, out_name)
    summ_results.to_csv(pd_file_name)


if __name__ == '__main__':
    main()
