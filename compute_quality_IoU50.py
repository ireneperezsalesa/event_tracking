# Compute precision and recall metrics for detection with YOLO on conventional and reconstructed frames.
# Plot found detections of the relevant class for a temporal sequence.
# Input: files produced by yolo_save_dets.py and groundtruth file from the dataset.


import numpy
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input_gt",
        help="Path to input file containing groundtruth bbox annotations",
    )
    parser.add_argument(
        "--absent_labels",
        default= None,
        help="Path to file containing absent labels",
    )
    parser.add_argument(
        "--input_rec",
        help="Path to input file containing predicted bboxes on reconstructed frames",
    )
    parser.add_argument(
        "--input_img",
        help="Path to input file containing predicted bboxes on conventional frames",
    )
    parser.add_argument(
        "--input_timestamps",
        help="Path to input file containing reconstruction timestamps",
    )
    parser.add_argument(
        "--det_class",
        help="Path to input file containing reconstruction timestamps",
    )

    return parser


def get_iou(bb1, bb2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Read absent labels file
    absent_labels = []
    if args.absent_labels is not None:
        with open(args.absent_labels) as file_absent:
            for line in file_absent:
                absent_labels.append(int(line))

    # Read groundtruth file
    with open(args.input_gt, "r") as file_gt:
        x1_gt = []
        y1_gt = []
        x2_gt = []
        y2_gt = []
        bboxes_gt = []
        center_gt_x = []
        k = 0
        for line in file_gt:
            if args.absent_labels is None or absent_labels[k] != 0:
                x, y, w, h = line.split(',')
                xmin, ymin, w, h = float(x), float(y), float(w), float(h)
                xmax = xmin + w
                ymax = ymin + h
                x1_gt.append(xmin)
                y1_gt.append(ymin)
                x2_gt.append(xmax)
                y2_gt.append(ymax)
                bbox = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax}
                bboxes_gt.append(bbox)
                center_gt_x.append((xmin + xmax) / 2)
            k = k + 1

    if args.absent_labels is None:
        absent_labels = []
        for i in range(len(bboxes_gt)):
            absent_labels.append(1)

    # Read file containing bboxes on reconstructed frames
    with open(args.input_rec, "r") as file_rec:
        img_rec = []
        bboxes_rec = []
        center_rec_x = []
        last_conf = None
        for line in file_rec:
            img, xmin, ymin, xmax, ymax, conf, clase = line.split(' ')
            img, xmin, ymin, xmax, ymax, conf, clase = int(img), float(xmin), float(ymin), float(xmax), float(ymax), float(conf), int(clase)
            bbox = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax, 'img': img}
            if clase == int(args.det_class): # 0 = person, 15 = cat
                # Keep center points of detected bounding box with most confidence for each frame - just for plot
                if last_conf is not None and img == bboxes_rec[-1]['img']:
                    if conf > last_conf:
                        center_rec_x[-1] = (xmin + xmax) / 2
                        last_conf = conf
                else:
                    center_rec_x.append((xmin + xmax) / 2)
                    last_conf = conf
                # Keep all detections for quality evaluation
                bboxes_rec.append(bbox)
                img_rec.append(img)

    # Read file containing bboxes on conventional frames
    with open(args.input_img, "r") as file_img:
        bboxes_img = []
        center_img_x = []
        last_conf = None
        for line in file_img:
            img, xmin, ymin, xmax, ymax, conf, clase = line.split(' ')
            img, xmin, ymin, xmax, ymax, conf, clase = int(img), float(xmin), float(ymin), float(xmax), float(ymax), float(conf), int(clase)
            bbox = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax, 'img': img}
            if clase == int(args.det_class):
                if last_conf is not None and img == bboxes_rec[-1]['img']:
                    if conf > last_conf:
                        center_img_x[-1] = (xmin + xmax) / 2
                        last_conf = conf
                else:
                    center_img_x.append((xmin + xmax) / 2)
                    last_conf = conf
                bboxes_img.append(bbox)


    # Read reconstruction timestamps file and compute timestamps for conventional frames
    with open(args.input_timestamps, "r") as file_ts:
        rec_ts = []
        for line in file_ts:
            ts = float(line)
            rec_ts.append(ts)

    start_ts = rec_ts[0]
    for i in range(len(rec_ts)):
        rec_ts[i] = (rec_ts[i] - start_ts)

    num_vis_img = len(absent_labels) - 1
    len_seq = rec_ts[-1]
    int_imgs = float(len_seq) / float(num_vis_img)
    t_gt_aux = numpy.arange(0., int_imgs * len(absent_labels), int_imgs)

    # Remove timestamps from groundtruth images with no label
    t_gt = []
    for i in range(len(t_gt_aux)):
        if absent_labels[i] != 0:
            t_gt.append(t_gt_aux[i])

    t_img = [int_imgs*i for i in range(len(absent_labels))]

    ts_rec_center = []
    last_img = -1

    for i in range(len(bboxes_rec)):
        img = bboxes_rec[i]['img']
        if img != last_img:
            ts_rec_center.append(rec_ts[img-1])
            last_img = img

    ts_img_center = []
    last_img = -1
    for i in range(len(bboxes_img)):
        img = bboxes_img[i]['img']
        if img != last_img:
            ts_img_center.append(t_img[img-1])
            last_img = img

    # Create interpolation functions for the 4 corners of the gt bounding boxes
    fx1 = interp1d(t_gt, x1_gt)
    fy1 = interp1d(t_gt, y1_gt)
    fx2 = interp1d(t_gt, x2_gt)
    fy2 = interp1d(t_gt, y2_gt)

    tmax = 20 #rec_ts[-1], max time to take into consideration for the sequence

    threshold = 50 # threshold for IoU eval

    # Compute true positives, false positives and false negatives of YOLO on conventional frames
    fp_img = 0
    tp_img = 0
    fn_img = 0
    bbox_gt = {}
    for i in range(len(t_img)):
        if t_img[i] < tmax:  # t_gt[-1]:
            # For every reconstructed frame, find the interpolated bbox corners
            bbox_gt['x1'] = fx1(t_img[i])
            bbox_gt['y1'] = fy1(t_img[i])
            bbox_gt['x2'] = fx2(t_img[i])
            bbox_gt['y2'] = fy2(t_img[i])

            # Check if the reconstruction has detections
            no_detections_for_img = True
            for n in range(len(bboxes_img)):
                if bboxes_img[n]['img'] == i:
                    # If there are detections on the reconstructed image, compare them to gt bbox
                    no_detections_for_img = False
                    iou = get_iou(bbox_gt, bboxes_img[n])
                    if iou * 100 < threshold:
                        fp_img = fp_img + 1
                    else:
                        tp_img = tp_img + 1
            # If there are no detections, false negative
            if no_detections_for_img == True:
                fn_img = fn_img + 1

    #print('On frames: TP: ', tp_img, ', FP: ', fp_img, ', FN: ', fn_img)
    precision_img = tp_img / (tp_img + fp_img)
    recall_img = tp_img / (tp_img + fn_img)


    # Compute true positives, false positives and false negatives of YOLO on reconstructed frames
    fp_rec = 0
    tp_rec = 0
    fn_rec = 0
    bbox_gt = {}
    for i in range(len(rec_ts)):
        if rec_ts[i] <= tmax:  # t_gt[-1]:
            # For every reconstructed frame, find the interpolated bbox corners
            bbox_gt['x1'] = fx1(rec_ts[i])
            bbox_gt['y1'] = fy1(rec_ts[i])
            bbox_gt['x2'] = fx2(rec_ts[i])
            bbox_gt['y2'] = fy2(rec_ts[i])

            # Check if the reconstruction has detections
            no_detections_for_img = True
            for n in range(len(bboxes_rec)):
                if bboxes_rec[n]['img'] == i:
                    # If there are detections on the reconstructed image, compare them to gt bbox
                    no_detections_for_img = False
                    iou = get_iou(bbox_gt, bboxes_rec[n])
                    if iou * 100 < threshold:
                        fp_rec = fp_rec + 1
                    else:
                        tp_rec = tp_rec + 1
            # If there are no detections, false negative
            if no_detections_for_img == True:
                fn_rec = fn_rec + 1

    #print('On rec: TP: ', tp_rec, ', FP: ', fp_rec, ', FN: ', fn_rec)
    if tp_rec or fp_rec > 0:
        precision_rec = tp_rec / (tp_rec + fp_rec)
    else:
        precision_rec = 0
    if tp_rec or fn_rec > 0:
        recall_rec = tp_rec / (tp_rec + fn_rec)
    else:
        recall_rec = 0


    # Number of frames for the evaluated sequence
    k_img = 0
    for i in range(len(t_img)):
        if t_img[i] <= tmax:
            k_img = k_img + 1

    k_rec = len(rec_ts)


    print('CONV FRAMES Precision at 50% threshold (%): ', precision_img*100)
    print('CONV FRAMES Recall at 50% threshold (%): ', recall_img*100)
    print('REC FRAMES Precision at 50% threshold (%): ', precision_rec*100)
    print('REC FRAMES Recall at 50% threshold (%): ', recall_rec*100)

    print('CONV FRAMES Images with detection vs total number of images (%): ', (k_img - fn_img)*100 / k_img)
    print('REC FRAMES Images with detection vs total number of images (%): ', (k_rec - fn_rec)*100 / k_rec)


    # Plot detections for a sequence - only most confident detection on each frame
    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.scatter(t_gt, center_gt_x, color='g', label='Groundtruth', s=3)
    plt.plot(t_gt, (fx1(t_gt) + fx2(t_gt)) / 2, color='g')
    plt.scatter(ts_img_center, center_img_x[0:len(ts_img_center)], color='r', label='Baseline', s=3)
    plt.scatter(ts_rec_center, center_rec_x, color='b', label='Our proposal', s=3)
    plt.xlabel('Timestamps (s)')
    plt.ylabel('X-coordinate (px)')
    plt.xlim([0, 20]) #ts_rec_center[-1]])
    plt.ylim([0, 300])
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('dets_pos_x.pdf', dpi=300, transparent=False, bbox_inches='tight')



