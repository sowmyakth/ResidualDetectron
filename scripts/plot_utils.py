"""Useful functions for plotting and summaring results"""
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scarlet
import scarlet.display
ROOT_DIR = '/home/users/sowmyak/ResidualDetectron'
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib


def plot_images(results, blend_image, val_image, val_cat, indx):
    center = val_cat.at[indx, 'scrlt_x0'], val_cat.at[indx, 'scrlt_y0']
    diff_image = np.transpose(val_image[indx, :, :, 1:4], axes=(2, 1, 0))
    model_image = np.transpose(val_image[indx, :, :, 7:10], axes=(2, 1, 0))
    image = np.transpose(blend_image[indx, :, :, 1:4], axes=(2, 1, 0))
    norm = scarlet.display.Asinh(img=model_image, Q=22)
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 4, 4)
    plt.plot(center[1], center[0], "go", mew=1)
    plt.plot(val_cat['gal1_x_tru'][indx], val_cat['gal1_y_tru'][indx], "rx",
             mew=1)
    plt.plot(val_cat['gal2_x_tru'][indx], val_cat['gal2_y_tru'][indx], "bx",
             mew=1)
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    norm_bl = scarlet.display.Asinh(img=image, Q=22)
    img_rgb = scarlet.display.img_to_rgb(image, norm=norm_bl)
    plt.imshow(img_rgb)
    plt.title('Blend Image')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.subplot(1, 4, 3)
    plt.plot(center[1], center[0], "go", mew=1)
    plt.plot(val_cat['gal1_x_tru'][indx], val_cat['gal1_y_tru'][indx], "rx",
             mew=1)
    plt.plot(val_cat['gal2_x_tru'][indx], val_cat['gal2_y_tru'][indx], "bx",
             mew=1)
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    model_img_rgb = scarlet.display.img_to_rgb(model_image, norm=norm)
    plt.imshow(model_img_rgb)
    plt.title('Net input:Model Image')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.subplot(1, 4, 2)
    diff_image_rgb = scarlet.display.img_to_rgb(diff_image)
    plt.imshow(diff_image_rgb)
    plt.title('Net input:Residual Image')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.plot(center[1], center[0], "ro", mew=1)
    plt.plot(val_cat['gal1_x_tru'][indx], val_cat['gal1_y_tru'][indx],
             "rx", mew=1)
    plt.plot(val_cat['gal2_x_tru'][indx], val_cat['gal2_y_tru'][indx],
             "bx", mew=1)
    plt.subplot(1, 4, 1)
    diff_image_rgb = scarlet.display.img_to_rgb(diff_image)
    plt.imshow(diff_image_rgb)
    plt.title('Net output:bbox(black)')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.plot(center[1], center[0], "ro", mew=1)
    x0, y0 = val_cat.at[indx, 'detect_x'], val_cat.at[indx, 'detect_y']
    h = val_cat.at[indx, 'detect_h']
    rect = patches.Rectangle((x0, y0),
                             h, h,
                             fill=False, color='yellow', alpha=0.7)
    ax = plt.gca()
    ax.add_patch(rect)
    for i in range(len(results[0]['scores'])):
        houtx = results[0]['rois'][i][3] - results[0]['rois'][i][1]
        houty = results[0]['rois'][i][2] - results[0]['rois'][i][0]
        rect = patches.Rectangle((results[0]['rois'][i][1],
                                  results[0]['rois'][i][0]),
                                 houtx, houty,
                                 fill=False, color='black', alpha=1)
        plt.text(results[0]['rois'][i][3], results[0]['rois'][i][2],
                 str(results[0]['scores'][i]),
                 color='black', size=12)
        ax = plt.gca()
        ax.add_patch(rect)
    plt.show()


def get_stats(ids, model, inference_config, dataset_val):
    names = ('unrecog_blnd', 'iter_recog', 'check',
             'iter_spurious2', 'iter_shred')
    vals = ('id', 'score', 'iou', 'dx', 'dy', 'ds')
    results = {}
    for name in names:
        results[name] = {}
        for val in vals:
            results[name][val] = []
    for image_id in ids:
        clss = 'check'
        original_image, image_meta, gt_class_id, gt_bbox=\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        result = model.detect([original_image])
        if len(result[0]['scores']) == 0:
            results['unrecog_blnd']['id'].append(image_id)
            continue
        gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox,
            gt_class_id, result[0]['rois'], result[0]['class_ids'],
            result[0]['scores'],
            iou_threshold=0., score_threshold=0.0)
        if len(result[0]['scores']) == 1:
            clss = 'iter_recog'
        else:
            if np.all(overlaps[0] > 0):
                clss = 'iter_shred'
            else:
                clss = 'iter_spurious2'
        gt_y0 = (gt_bbox[:, 2] + gt_bbox[:, 0]) / 2.
        gt_x0 = (gt_bbox[:, 3] + gt_bbox[:, 1]) / 2.
        pred_x0 = (result[0]['rois'][:, 3] + result[0]['rois'][:, 1]) / 2.
        pred_y0 = (result[0]['rois'][:, 2] + result[0]['rois'][:, 0]) / 2.
        dx = gt_x0 - pred_x0
        dy = gt_y0 - pred_y0
        ds = np.hypot(dx, dy) / dataset_val.Y['distance'][image_id] * 0.2
        for j in range(len(result[0]['scores'])):
            results[clss]['id'].append(image_id)
            results[clss]['score'].append(result[0]['scores'][j])
            results[clss]['iou'].append(overlaps[j][0])
            results[clss]['dx'].append(dx[j])
            results[clss]['dy'].append(dy[j])
            results[clss]['ds'].append(ds[j])
    return results
