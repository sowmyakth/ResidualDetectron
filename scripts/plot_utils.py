"""Useful functions for plotting and summaring results"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scarlet
import scarlet.display
import btk


def plot_input(blend_image, val_image, val_cat, indx):
    center = val_cat.at[indx, 'scrlt_x0'], val_cat.at[indx, 'scrlt_y0']
    diff_image = np.transpose(val_image[indx, :, :, 1:4], axes=(2, 1, 0))
    model_image = np.transpose(val_image[indx, :, :, 7:10], axes=(2, 1, 0))
    image = np.transpose(blend_image[indx, :, :, 1:4], axes=(2, 1, 0))
    norm = scarlet.display.Asinh(img=model_image, Q=22)
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 3)
    plt.plot(center[1], center[0], "go", mew=1, markersize=8)
    plt.plot(val_cat['gal1_x_tru'][indx], val_cat['gal1_y_tru'][indx], "rx",
             mew=1)
    plt.plot(val_cat['gal2_x_tru'][indx], val_cat['gal2_y_tru'][indx], "bx",
             mew=1)
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    if val_cat['gal1_x_stck'][indx] == -1:
        plt.plot(val_cat['gal2_y_stck'][indx], val_cat['gal2_x_stck'][indx],
                 "c*", mew=1, markersize=12)
    else:
        plt.plot(val_cat['gal1_y_stck'][indx], val_cat['gal1_x_stck'][indx],
                 "c*", mew=1, markersize=12)
    norm_bl = scarlet.display.Asinh(img=image, Q=22)
    img_rgb = scarlet.display.img_to_rgb(image, norm=norm_bl)
    plt.imshow(img_rgb)
    plt.title('Blend Image')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.subplot(1, 3, 2)
    plt.plot(center[1], center[0], "go", mew=1, markersize=8)
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
    plt.subplot(1, 3, 1)
    diff_image_rgb = scarlet.display.img_to_rgb(diff_image)
    plt.imshow(diff_image_rgb)
    plt.title('Net input:Residual Image')
    plt.xlim([30, 90])
    plt.ylim([30, 90])
    plt.plot(center[1], center[0], "go", mew=1)
    plt.plot(val_cat['gal1_x_tru'][indx], val_cat['gal1_y_tru'][indx],
             "rx", mew=1)
    plt.plot(val_cat['gal2_x_tru'][indx], val_cat['gal2_y_tru'][indx],
             "bx", mew=1)
    x0, y0 = val_cat.at[indx, 'detect_x'], val_cat.at[indx, 'detect_y']
    h = val_cat.at[indx, 'detect_h']
    rect = patches.Rectangle((x0, y0),
                             h, h,
                             fill=False, color='yellow', alpha=0.7)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.show()


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
    plt.plot(center[1], center[0], "go", mew=1)
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
    plt.plot(center[1], center[0], "go", mew=1)
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


def plot_results_images(results, blend_image, val_image, val_cat, indx):
    q, = np.where(results['id'] == indx)
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
    plt.plot(center[1], center[0], "go", mew=1)
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
    plt.plot(center[1], center[0], "go", mew=1)
    x0, y0 = val_cat.at[indx, 'detect_x'], val_cat.at[indx, 'detect_y']
    h = val_cat.at[indx, 'detect_h']
    rect = patches.Rectangle((x0, y0),
                             h, h,
                             fill=False, color='yellow', alpha=0.7)
    ax = plt.gca()
    ax.add_patch(rect)
    for i in q:
        houtx = results['roi'][i][3] - results['roi'][i][1]
        houty = results['roi'][i][2] - results['roi'][i][0]
        rect = patches.Rectangle((results['roi'][i][1],
                                  results['roi'][i][0]),
                                 houtx, houty,
                                 fill=False, color='black', alpha=1)
        plt.text(results['roi'][i][3], results['roi'][i][2],
                 str(results['score'][i]),
                 color='black', size=12)
        ax = plt.gca()
        ax.add_patch(rect)
    plt.show()

def get_tru_pred_box(indx, results, val_cat, ax):
    x0, y0 = val_cat.at[indx, 'detect_x'], val_cat.at[indx, 'detect_y']
    h = val_cat.at[indx, 'detect_h']
    rect = patches.Rectangle((x0, y0),
                             h, h,
                             fill=False, color='yellow', alpha=0.7)
    ax = plt.gca()
    ax.add_patch(rect)
    for i in q:
        if results['roi_x1'][i] < 20:
            continue
        houtx = results['roi_x2'][i] - results['roi_x1'][i]
        houty = results['roi_y2'][i] - results['roi_y1'][i]
        rect = patches.Rectangle((results['roi_x1'][i],
                                  results['roi_y1'][i]),
                                 houtx, houty,
                                 fill=False, color='black', alpha=1)
        plt.text(results['roi_x2'][i], results['roi_y2'][i],
                 str(results['network_score'][i]),
                 color='black', size=12)
        ax = plt.gca()
        ax.add_patch(rect)


def plot_metrics_images(results, blend_image, val_image, val_cat, indx):
    q, = np.where(results['id'] == indx)
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
    plt.plot(center[1], center[0], "go", mew=1)
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
    plt.plot(center[1], center[0], "go", mew=1)

    plt.show()


def plot_hist(full_cat, Y, indxs):
    plt.figure(figsize=(13, 3))
    plt.subplot(1, 3, 1)
    plt.scatter(full_cat['ab_mag_1'][indxs],
                full_cat['ab_mag_2'][indxs], alpha=0.1)
    plt.xlabel('mag of gal1')
    plt.ylabel('mag of gal2')
    plt.subplot(1, 3, 2)
    plt.hist(Y['distance'][indxs] / 0.2)
    plt.xlabel('distance between gal pair (pixels)')
    plt.subplot(1, 3, 3)
    dist = np.hypot(Y['detect_x'] + Y['detect_h'] / 2. - Y['scrlt_y0'],
                    Y['detect_y'] + Y['detect_h'] / 2. - Y['scrlt_x0'])[indxs]
    plt.title('dist btwn scarlet center & undet gal/ dist btwn gal pair ')
    plt.hist(dist / Y['distance'][indxs] * 0.2, np.linspace(0.45, 1, 20))


def plot_iter_detections(blend_images, blend_list, iter_detected_centers,
                         detected_centers=None, limits=None,
                         band_indices=[1, 2, 3]):
    """Plots blend images as RGB image, sum in all bands, and RGB image with
    centers of objects marked.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects. If detected_centers are input, then the centers are also
    shown in the third panel along with the true centers.

    Args:
        blend_images (array_like): Array of blend scene images to plot
            [batch, height, width, bands].
        blend_list (list) : List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        detected_centers (list, default=`None`): List of `numpy.ndarray` or
            lists with centers of detected centers for each image in batch.
            Length of list must be the batch size. Each list entry must be a
            list or `numpy.ndarray` of dimensions [N, 2].
        limits(list, default=`None`): List of start and end coordinates to
            display image within. Note: limits are applied to both height and
            width dimensions.
        band_indices (list, default=[1,2,3]): list of length 3 with indices of
            bands that are to be plotted in the RGB image.
    """
    batch_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError("band_indices must be a list with 3 entries, not",
                         f"{band_indices}")
    if detected_centers is None:
        detected_centers = [[]] * batch_size
    if (len(detected_centers) != batch_size or
            blend_images.shape[0] != batch_size):
        raise ValueError(f"Length of detected_centers and length of blend_list\
            must be equal to first dimension of blend_images, found \
            {len(detected_centers), len(blend_list), len(blend_images)}")
    for i in range(batch_size):
        num = len(blend_list[i])
        itr_num = len
        images = np.transpose(blend_images[i],
                              axes=(2, 0, 1))
        blend_img_rgb = btk.plot_utils.get_rgb_image(images[band_indices])
        _, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(blend_img_rgb)
        if limits:
            ax[0].set_xlim(limits)
            ax[0].set_ylim(limits)
        ax[0].set_title("gri bands")
        ax[0].axis('off')
        ax[1].imshow(np.sum(blend_images[i, :, :, :], axis=2))
        ax[1].set_title("Sum")
        if limits:
            ax[1].set_xlim(limits)
            ax[1].set_ylim(limits)
        ax[1].axis('off')
        ax[2].imshow(blend_img_rgb)
        iter_num = len(iter_detected_centers[i])
        ax[2].set_title(f"{itr_num} iter detecetd")
        for entry in blend_list[i]:
            ax[2].plot(entry['dx'], entry['dy'], 'rx')
        if limits:
            ax[2].set_xlim(limits)
            ax[2].set_ylim(limits)
        for cent in detected_centers[i]:
            ax[2].plot(cent[0], cent[1], 'go', fillstyle='none', ms=10, mew=2)
        for cent in iter_detected_centers[i]:
            ax[2].plot(cent[0], cent[1], 'yo', fillstyle='none', ms=10, mew=2)
        ax[2].axis('off')
    plt.show()
