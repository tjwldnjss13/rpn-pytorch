import numpy as np

from utils.util import calculate_ious
from pytorch_util import calculate_iou


def anchor_generator(feature_size, anchor_stride):
    # Inputs:
    #    feature_size: (h, w)
    #    anchor_stride: Int
    # Outputs:
    #    anchors: numpy array [-1, (cy, cx)]

    h_f, w_f = feature_size
    xs_ctr = np.arange(anchor_stride, (w_f + 1) * anchor_stride, anchor_stride)
    ys_ctr = np.arange(anchor_stride, (h_f + 1) * anchor_stride, anchor_stride)

    anchors = np.zeros((len(xs_ctr) * len(ys_ctr), 2))

    c_i = 0
    for x in xs_ctr:
        for y in ys_ctr:
            anchors[c_i, 1] = x - anchor_stride // 2
            anchors[c_i, 0] = y - anchor_stride // 2
            c_i += 1

    print('Anchors generated')

    return anchors


def anchor_box_generator(ratios, scales, input_size, anchor_stride):
    # 50 = 800 // 16

    # ratios = [.5, 1, 2]
    # scales = [128, 256, 512]

    in_h, in_w = input_size[0], input_size[1]

    feat_h, feat_w = in_h // anchor_stride, in_w // anchor_stride
    anchors = anchor_generator((feat_h, feat_w), anchor_stride)
    anchor_boxes = np.zeros((len(anchors) * len(ratios) * len(scales), 4))

    anc_i = 0
    for anc in anchors:
        anc_y, anc_x = anc
        for r in ratios:
            for s in scales:
                # h = anchor_stride * s * np.sqrt(r)
                # w = anchor_stride * s * np.sqrt(1. / r)

                # if r < 1:
                #     h, w = s, s * (1. / r)
                # elif r > 1:
                #     h, w = s * (1. / r), s
                # else:
                #     h, w = s, s

                if r < 1:
                    h, w = s / r, s
                elif r > 1:
                    h, w = s, s * r
                else:
                    h, w = s, s

                anchor_boxes[anc_i, 0] = anc_y - .5 * h
                anchor_boxes[anc_i, 1] = anc_x - .5 * w
                anchor_boxes[anc_i, 2] = anc_y + .5 * h
                anchor_boxes[anc_i, 3] = anc_x + .5 * w

                # anchor_boxes[anc_i, 0] = anc_y
                # anchor_boxes[anc_i, 1] = anc_x
                # anchor_boxes[anc_i, 2] = h
                # anchor_boxes[anc_i, 3] = w

                anc_i += 1

    # idx_valid = np.where((anchor_boxes[:, 0] >= 0) &
    #                      (anchor_boxes[:, 1] >= 0) &
    #                      (anchor_boxes[:, 2] <= in_h) &
    #                      (anchor_boxes[:, 3] <= in_w))[0]
    # anchor_boxes = anchor_boxes[idx_valid]

    print('Anchor boxes generated')

    return anchor_boxes


import torch
anc_boxes = anchor_box_generator([.5, 1, 2], [32, 64, 128], (224, 224), 16)
anc_boxes = torch.as_tensor(anc_boxes)
anc_boxes = anc_boxes.reshape(14, 14, 36)
# anc_boxes = anc_boxes.permute(2, 0, 1)
print(anc_boxes[:10, 0, 0])


def anchor_boxes_generator_categorical(anchor_boxes, ground_truth):
    n_gt = ground_truth.shape[0]
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_boxes_cat = [[] for _ in range(n_gt)]
    for i, arg in enumerate(argmax_iou_anc_gt):
        anchor_boxes_cat[arg].append(anchor_boxes[i])

    # anchor_gts = ground_truth[argmax_iou_anc_gt]

    for i in range(n_gt):
        anchor_boxes_cat[i] = np.array(anchor_boxes_cat[i])
    # anchor_boxes_cat = np.array(anchor_boxes_cat)

    print('Categorical anchor boxes generated')

    return anchor_boxes_cat


def anchor_label_generator(anchor_boxes, ground_truth, pos_threshold, neg_threshold):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)

    pos_args_ious_anc_gt_1 = np.argmax(ious_anc_gt, axis=0)
    pos_args_ious_anc_gt_2 = np.where(ious_anc_gt >= pos_threshold)[0]
    pos_args_ious_anc_gt = np.append(pos_args_ious_anc_gt_1, pos_args_ious_anc_gt_2)
    pos_args_ious_anc_gt = np.array(list(set(pos_args_ious_anc_gt)))

    # anchor_labels = np.zeros(anchors.shape[0])
    anchor_labels = np.array([-1 for _ in range(anchor_boxes.shape[0])])
    anchor_labels[pos_args_ious_anc_gt] = 1

    non_pos_args_labels = np.where(anchor_labels != 1)[0]
    for i in non_pos_args_labels:
        neg_f = False
        for j in range(len(ground_truth)):
            if ious_anc_gt[i, j] >= neg_threshold:
                break
            neg_f = True
        if neg_f:
            anchor_labels[i] = 0

    # neg_args_ious_anc_gt = np.where(anchor_labels == -1)[0]

    print('Anchor labels generated')

    return anchor_labels


def anchor_label_generatgor_2dim(anchor_labels):
    anchor_labels2 = np.zeros((anchor_labels.shape[0], 2))
    train_args = np.where(anchor_labels != -1)
    anchor_labels2[train_args, anchor_labels[train_args]] = 1

    print('2-dim anchor labels generated')

    return anchor_labels2


def anchor_ground_truth_generator(anchor_boxes, ground_truth):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_gts = ground_truth[argmax_iou_anc_gt]

    print('Anchor ground truth generated')

    return anchor_gts


def loc_delta_generator(bbox, anchor_box):
    # Inputs:
    #    bbox: tensor [-1, (y1, x1, y2, x2)]
    #    anchor_box: tensor [-1, (y1, x1, y2, x2)]
    # Outputs:
    #    locs: tensor [-1, (cy, cx, h, w)]

    assert bbox.shape == anchor_box.shape

    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_cy, bbox_cx = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    anc_h, anc_w = anchor_box[:, 2] - anchor_box[:, 0], anchor_box[:, 3] - anchor_box[:, 1]
    anc_cy, anc_cx = anchor_box[:, 0] + .5 * anc_h, anchor_box[:, 1] + .5 * anc_w

    loc_h, loc_w = torch.log(bbox_h / anc_h), torch.log(bbox_w / anc_w)
    loc_cy, loc_cx = (bbox_cy - anc_cy) / anc_h, (bbox_cx - anc_cx) / anc_w

    locs = torch.zeros(bbox.shape)
    locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3] = loc_cy, loc_cx, loc_h, loc_w

    return locs


def non_maximum_suppression(bbox, score, threshold=.5):
    # Inputs:
    #    bbox: tensor [num_batch, -1, (y1, x1, y2, x2)]
    #    score: tensor [num_batch, n]
    #    threshold: Float
    # Outputs:
    #    bbox_nms: tensor [num_batch, -1, (y1, x1, y2, x2)]
    #    score_nms: tensor [num_batch, n]

    keep = torch.ones(score.shape)
    v, idx = score.sort(descending=True)
    bbox_base = bbox[idx[0]]

    for b in range(bbox.shape[0]):
        for i in range(1, len(idx)):
            bbox_temp = bbox[idx[i]]
            iou = calculate_iou(bbox_base, bbox_temp)
            if iou > threshold:
                keep[i] = 0

    return bbox * keep.reshape(len(keep), -1), score * keep


def generate_rpn_target(bboxes, scores, anchor_boxes, image_size, in_size, out_size):
    # Inputs:
    #    bboxes: tensor [-1, (y1, x1, y2, x2)]
    #    scores: tensor [-1, (no obj score, obj score)]
    #    anchor_boxes: tensor [output height, output width, 4 * 9]
    #    image_size: (depth, image height, image width)
    #    in_size: (input height, input width)
    #    out_size: (output height, output width)

    target_bbox = torch.zeros(anchor_boxes.shape)
    ratio_h, ratio_w = in_size[0] / image_size[1], in_size[1] / image_size[2]

    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        y1, x1, y2, x2 = y1 * ratio_h, x1 * ratio_w, y2 * ratio_h, x2 * ratio_w
        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        h, w = y2 - y1, x2 - x1
        cy_idx, cx_idx = (cy * out_size[0] / in_size[0]).int(), (cx * out_size[1] / in_size[1]).int()
        bbox_tensor = torch.Tensor([cy, cx, h, w])

        idx_max = 0
        iou_max = 0
        for i in range(9):
            iou_temp = calculate_iou(bbox_tensor, anchor_boxes[cy_idx, cx_idx, 4 * i:4 * (i + 1)])
            if iou_temp > iou_max:
                idx_max = i
                iou_max = iou_temp

        target_bbox[cy, cx, 4 * idx_max:4 * (idx_max + 1)] = bbox_tensor






    






























