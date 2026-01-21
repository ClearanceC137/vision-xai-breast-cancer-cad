# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:36:08 2025

@author: clear
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_backprops_for_explainability(model, y_true):
    grads = [None] * len(model.layers)
    d_out = None
    conv_act_grads = {}

    for r_idx, layer in enumerate(reversed(model.layers)):
        idx = len(model.layers) - 1 - r_idx
        if layer['type'] == 'output':
            probs = model._softmax(layer['z'])
            d_out = probs - y_true
            dW = np.outer(d_out, layer['input'])
            db = d_out.copy()
            grads[idx] = {'dW': dW, 'db': db}
            d_out = np.dot(layer['weights'].T, d_out)

        elif layer['type'] == 'dense':
            d_activation = np.where(layer['z'] > 0, 1.0, model.leaky_alpha)
            dz = d_out * d_activation
            dW = np.outer(dz, layer['input'])
            db = dz.copy()
            grads[idx] = {'dW': dW, 'db': db}
            d_out = np.dot(layer['weights'].T, dz)

        elif layer['type'] == 'pool':
            if d_out.ndim == 1:
                out_h, out_w, C = layer['output_shape']
                d_out = d_out.reshape((out_h, out_w, C))
            dX = model._max_pool_backward(d_out, layer['switches'], layer['input'].shape)
            d_out = dX
            grads[idx] = None

        elif layer['type'] == 'conv':
            x = layer['input']
            k = layer['ksize']
            H, W, F = layer['output_shape']
            dX = np.zeros_like(x)
            dF_all = np.zeros_like(layer['filters'])
            db_all = np.zeros_like(layer['biases'])
            for f in range(F):
                grad_accum = np.zeros_like(layer['filters'][f])
                for i in range(H):
                    for j in range(W):
                        local_mask = 1.0 if layer['output'][i, j, f] > 0 else model.leaky_alpha
                        grad_val = local_mask * d_out[i, j, f]
                        for c in range(x.shape[2]):
                            patch = x[i:i+k, j:j+k, c]
                            grad_accum[:, :, c] += grad_val * patch
                            dX[i:i+k, j:j+k, c] += layer['filters'][f][:, :, c] * grad_val
                dF_all[f] = grad_accum
                db_all[f] = np.sum(np.where(layer['output'][:, :, f] > 0, 1.0, model.leaky_alpha) * d_out[:, :, f])
            grads[idx] = {'dF': dF_all, 'db_conv': db_all}
            conv_act_grads[idx] = d_out.copy()
            d_out = dX

    d_input = d_out
    return grads, d_input, conv_act_grads


def generate_saliency_overlay(img, d_input):
    saliency = np.abs(d_input).max(axis=-1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(saliency * 255)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img.astype(np.uint8), 0.5, heatmap, 0.5, 0)
    return overlay, heatmap


def generate_dual_class_overlays(model, img, classes_to_test=[0, 1], save_folder='explainability'):
    # Ensure folder exists
    os.makedirs(save_folder, exist_ok=True)

    overlays = {}
    for class_idx in classes_to_test:
        y_true = np.zeros(model.layers[-1]['biases'].shape, dtype=np.float32)
        y_true[class_idx] = 1.0

        # forward pass
        model.forward(img, training=False)

        # compute gradients
        grads, d_input, conv_act_grads = compute_backprops_for_explainability(model, y_true)

        # generate overlay
        overlay, heatmap = generate_saliency_overlay(img, d_input)

        # save images
        overlay_path = os.path.join(save_folder, f'overlay_class_{class_idx}.png')
        heatmap_path = os.path.join(save_folder, f'heatmap_class_{class_idx}.png')
        cv2.imwrite(overlay_path, overlay)
        cv2.imwrite(heatmap_path, heatmap)

        overlays[class_idx] = (overlay, heatmap)
        print(f"Saved overlay and heatmap for class {class_idx} in {save_folder}")

    return overlays
