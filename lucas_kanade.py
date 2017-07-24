# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:25:07 2017

@author: ryuhei

Implementations of some variants of Lucas-Kanade algorithm from the paper:
Simon Baker and Iain Matthews, "Lucas-kanade 20 years on: A unifying framework"
, International journal of computer vision 56.3 (2004): 221-255.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_data():
    image = cv2.imread('kanade.jpg', False) / 255.0
    M = np.array([[1.0, 0.2, 50],
                  [-0.2, 1.0, 90]], dtype='d')
#    M = np.array([[1.0, 0.0, 50],
#                  [0.0, 1.0, 90]], dtype='d')
    M = cv2.invertAffineTransform(M)
    template = cv2.warpAffine(image, M, (50, 50))
    M = cv2.invertAffineTransform(M)
    return image, template, M


def show_data(image, template, mat_affine):
    h, w = template.shape
    a = np.array([[0, 0, w, w],
                  [0, h, h, 0],
                  [1, 1, 1, 1]])
    b = mat_affine.dot(a)
    plt.matshow(image, cmap=plt.cm.gray)
    plt.plot(b[0, [0, 1, 2, 3, 0]], b[1, [0, 1, 2, 3, 0]], '-or')
    plt.show()
    plt.matshow(template, cmap=plt.cm.gray)
    plt.show()


def grad_image(image, border=False):
    '''
    Compute gradient of image w.r.t. x and y by central difference method.
    Let W, H := width and height of image. Then this function returns gigx and
    gigy, where
    gigx[x, y] := (image[x + 1, y] - image[x - 1, y]) / 2
    for x in (1, ..., W - 2) and y in (0, ..., H - 1),
    gigy[x, y] := (image[x, y + 1] - image[x, y - 1]) / 2
    for x in (0, ..., W - 1) and y in (1, ..., H - 2).
    If border is True, then the left and right columns of gigx and the top and
    bottom rows of gigy are computed by forward or backward difference method.
    Otherwise, those border values are set to 0.
    '''
    gigx = np.zeros_like(image)
    gigx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    gigy = np.zeros_like(image)
    gigy[1:-1] = (image[2:] - image[:-2]) / 2
    if border:
        gigx[:, 0] = image[:, 1] - image[:, 0]
        gigx[:, -1] = image[:, -1] - image[:, -2]
        gigy[0] = image[1] - image[0]
        gigy[-1] = image[-1] - image[-2]
    return gigx, gigy


def lk_forward_additive(img, tmpl, p_init=None):
    '''Lucas-Kanade forward additive method (Fig. 1) for affine warp model.

    Args:
        img (numpy.ndarray):
            A grayscale image of shape (height, width).
        tmpl (numpy.ndarray):
            A template image (height_t, width_t).
        p_init (numpy.ndarray):
            Initial parameter of affine transform which shape is (2, 3).

    Returns:
        ps (list):
            Estimates of p for each iteration.
    '''

    # Warp function W(x; p) is a mapping from x_template to x_image
    height, width = img.shape
    gx_img, gy_img = grad_image(img)

    # initialize p
    if p_init is not None:
        p = p_init.copy()
    else:
        p = np.array([[1.2, 0.0, 50],
                      [0.0, 1.2, 70]], dtype='d')
    ps = [p.copy()]

    for it in range(max_iteration):
        p_inv = cv2.invertAffineTransform(p)
        # step (1)
        img_w = cv2.warpAffine(img, p_inv, tmpl.shape)

        # step (2)
        error = tmpl - img_w

        # step (3)
        gx_img_w = cv2.warpAffine(gx_img, p_inv, tmpl.shape)
        gy_img_w = cv2.warpAffine(gy_img, p_inv, tmpl.shape)

        # step (4)
        g_img_w_height, g_img_w_width = gx_img_w.shape
        x, y = np.meshgrid(np.arange(g_img_w_width), np.arange(g_img_w_height))
        zero = np.zeros_like(x)
        one = np.ones_like(x)
        gp_warp = np.array([[x, y, one, zero, zero, zero],
                            [zero, zero, zero, x, y, one]], dtype='d')

        # step (5)
        g_img_w = np.stack((gx_img_w, gy_img_w), axis=0)
        # compute the steepest descent images
        sd_imgs = np.einsum('jhw,jihw->ihw', g_img_w, gp_warp)

        # step (6)
        sd_imgs_flat = sd_imgs.reshape(6, -1)
        hessian = sd_imgs_flat.dot(sd_imgs_flat.T)

        # step (7)
        # steepest descent parameter update
        sd_updates = sd_imgs_flat.dot(error.ravel())

        # step (8)
        p_update = sd_updates.dot(np.linalg.inv(hessian)).reshape(p.shape)

        # step (9)
        p += p_update
        ps.append(p.copy())

        # converged?
        if np.linalg.norm(p_update) < convergence_threshold:
            print('Converged.')
            break
    return ps


def lk_forward_compositional(img, tmpl, p_init=None):
    '''Lucas-Kanade forward compositional method (Fig. 3) for affine warp.

    Args:
        img (numpy.ndarray):
            A grayscale image of shape (height, width).
        tmpl (numpy.ndarray):
            A template image (height_t, width_t).
        p_init (numpy.ndarray):
            Initial parameter of affine transform which shape is (2, 3).

    Returns:
        ps (list):
            Estimates of p for each iteration.
    '''

    # Warp function W(x; p) is a mapping from x_template to x_image
    height, width = img.shape
    gx_img, gy_img = grad_image(img)

    # initialize p
    if p_init is not None:
        p = p_init.copy()
    else:
        p = np.array([[1.2, 0.0, 50],
                      [0.0, 1.2, 70]], dtype='d')
    ps = [p.copy()]

    # precompute step (4)
    g_img_w_height, g_img_w_width = tmpl.shape
    x, y = np.meshgrid(np.arange(g_img_w_width), np.arange(g_img_w_height))
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    gp_warp = np.array([[x, y, one, zero, zero, zero],
                        [zero, zero, zero, x, y, one]], dtype='d')

    for it in range(max_iteration):
        p_inv = cv2.invertAffineTransform(p)
        # step (1)
        img_w = cv2.warpAffine(img, p_inv, tmpl.shape)

        # step (2)
        error = tmpl - img_w

        # step (3)
        # In order to compute gradient of a warped image by central difference,
        # the warped image have to be defined on [-1, W] \times [-1, H] (i.e.
        # it need extra pixels outside the margin), where W and H are the width
        # and the height of the template image.
        p_tmp = p.copy()
        offset = p_tmp.dot(np.array([[-1, -1, 0]]).T)
        p_tmp[:, 2:3] += offset
        p_inv_pad = cv2.invertAffineTransform(p_tmp)
        img_w_pad = cv2.warpAffine(img, p_inv_pad,
                                   (tmpl.shape[0]+2, tmpl.shape[1]+2))
        gx_img_w, gy_img_w = grad_image(img_w_pad)
        gx_img_w = gx_img_w[1:-1, 1:-1].copy()
        gy_img_w = gy_img_w[1:-1, 1:-1].copy()
#        gx_img_w, gy_img_w = grad_image(img_w, True)

        # step (5)
        g_img_w = np.stack((gx_img_w, gy_img_w), axis=0)
        # compute the steepest descent images
        sd_imgs = np.einsum('jhw,jihw->ihw', g_img_w, gp_warp)

        # step (6)
        sd_imgs_flat = sd_imgs.reshape(6, -1)
        hessian = sd_imgs_flat.dot(sd_imgs_flat.T)

        # step (7)
        # steepest descent parameter update
        sd_updates = sd_imgs_flat.dot(error.ravel())

        # step (8)
        p_update = sd_updates.dot(np.linalg.inv(hessian)).reshape(p.shape)

        # step (9)
        p_33 = np.vstack((p, [0, 0, 1]))
        p_update_33 = np.eye(3)
        p_update_33[:2] += p_update
        p = p_33.dot(p_update_33)[:2]
        ps.append(p.copy())

        # converged?
        if np.linalg.norm(p_update) < convergence_threshold:
            print('Converged.')
            break
    return ps


def lk_inverse_additive(img, tmpl, p_init=None):
    '''Lucas-Kanade inverse additive method (Fig. 5) for affine warp.

    Args:
        img (numpy.ndarray):
            A grayscale image of shape (height, width).
        tmpl (numpy.ndarray):
            A template image (height_t, width_t).
        p_init (numpy.ndarray):
            Initial parameter of affine transform which shape is (2, 3).

    Returns:
        ps (list):
            Estimates of p for each iteration.
    '''
    raise NotImplementedError


def lk_inverse_compositional(img, tmpl, p_init=None):
    '''Lucas-Kanade inverse compositional method (Fig. 4) for affine warp.

    Args:
        img (numpy.ndarray):
            A grayscale image of shape (height, width).
        tmpl (numpy.ndarray):
            A template image (height_t, width_t).
        p_init (numpy.ndarray):
            Initial parameter of affine transform which shape is (2, 3).

    Returns:
        ps (list):
            Estimates of p for each iteration.
    '''

    # Warp function W(x; p) is a mapping from x_template to x_image
    height, width = img.shape

    # initialize p
    if p_init is not None:
        p = p_init.copy()
    else:
        p = np.array([[1.2, 0.0, 50],
                      [0.0, 1.2, 70]], dtype='d')
    ps = [p.copy()]

    # precompute step (3)
    gx_tmpl, gy_tmpl = grad_image(tmpl, True)

    # precompute step (4)
    g_img_w_height, g_img_w_width = tmpl.shape
    x, y = np.meshgrid(np.arange(g_img_w_width), np.arange(g_img_w_height))
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    gp_warp = np.array([[x, y, one, zero, zero, zero],
                        [zero, zero, zero, x, y, one]], dtype='d')

    # precompute step (5)
    g_tmpl = np.stack((gx_tmpl, gy_tmpl), axis=0)
    # compute the steepest descent images
    sd_imgs = np.einsum('jhw,jihw->ihw', g_tmpl, gp_warp)

    # step (6)
    sd_imgs_flat = sd_imgs.reshape(6, -1)
    hessian = sd_imgs_flat.dot(sd_imgs_flat.T)

    for it in range(max_iteration):
        p_inv = cv2.invertAffineTransform(p)
        # step (1)
        img_w = cv2.warpAffine(img, p_inv, tmpl.shape)

        # step (2)
        error = img_w - tmpl

        # step (7)
        # steepest descent parameter update
        sd_updates = sd_imgs_flat.dot(error.ravel())

        # step (8)
        p_update = sd_updates.dot(np.linalg.inv(hessian)).reshape(p.shape)

        # step (9)
        p_33 = np.vstack((p, [0, 0, 1]))
        p_update_33 = np.eye(3)
        p_update_33[:2] += p_update
        p = p_33.dot(np.linalg.inv(p_update_33))[:2]
        ps.append(p.copy())

        # converged?
        if np.linalg.norm(p_update) < convergence_threshold:
            print('Converged.')
            break
    return ps


if __name__ == '__main__':
    convergence_threshold = 0.0001
    max_iteration = 200
    lk_methods = [lk_forward_additive, lk_forward_compositional,
                  lk_inverse_additive, lk_inverse_compositional]

    img, tmpl, M = get_data()
    tmpl_shape = tmpl.shape
    show_data(img, tmpl, M)

    p_init = np.array([[1.2, 0.0, 40],
                       [0.0, 1.2, 80]], dtype='d')
    ps = lk_methods[3](img, tmpl, p_init)

    # show the result
    print('Iterations:', len(ps))
#    for i, p in enumerate(ps):
    for i, p in enumerate([ps[0], ps[-1]]):
        p_inv = cv2.invertAffineTransform(p)
        img_w = cv2.warpAffine(img, p_inv, tmpl.shape)
        print('#', i)
        print('error:', np.linalg.norm(tmpl - img_w))
        print('p:')
        print(p)
        show_data(img, img_w, p)
