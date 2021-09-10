import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    pass


def convert_bbox_to_z(bbox):
    """
    Convert a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x, y is the center of the box and s is the scale/area and r
    is the aspect ratio
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert a bounding box in the center form [x, y, s, r] to itself in the form
    [x1, y1, x2, y2] where (x1, y1) is the top left and (x2, y2) is the bottom right
    :param x: [x, y, s, r]
    :param score: Optional
    :return: [x1, y1, x2, y2] if score is None else [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.,
        ])
    else:
        return np.array([
            x[0] - w / 2.,
            x[1] - h / 2.,
            x[0] + w / 2.,
            x[1] + h / 2.,
            score,
        ])
