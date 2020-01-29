"""Useful functions for plotting and summaring results"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scarlet
import scarlet.display


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
