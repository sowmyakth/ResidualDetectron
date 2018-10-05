"""Useful functions for plotting and summaring results"""
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scarlet
import scarlet.display


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
