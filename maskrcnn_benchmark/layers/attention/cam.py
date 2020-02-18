from torch.nn import functional as F
import numpy as np
import cv2


def ClassActivationMapping(feature_conv, weight_softmax, class_idx, size_upsample=(256, 256)):
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def visualize_cam(cam, root_img=None):
    if root_img is not None:
        img = cv2.imread(root_img)
        height, width, _ = img.shape
        cam = cv2.resize(cam, (width, height))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
    else:
        result = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return result


def get_weight_softmax(net, layer_name):
    # params = list(net.parameters())
    # weight_softmax = np.squeeze(params[layer_name].data.cpu().numpy())
    weight_softmax = net.layers[layer_name].get_weights()
    return weight_softmax


def get_cam(weight_softmax, features_blobs, logit, root_img=None, classes=None):
    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)

    if classes is not None:
        print('top1 prediction: %s' % classes[idx[0]])
        for i in range(idx):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    CAMs = ClassActivationMapping(features_blobs[0], weight_softmax, [idx[0]])

    results = []
    for i in idx:
        result = visualize_cam(CAMs[i], root_img)
        results.append(result)

    return result, idx


