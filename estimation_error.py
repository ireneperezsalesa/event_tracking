# Compute estimation error metrics for baseline tracker and our tracker.
# Input: txt files produced by estimation_event_tracker.py and estimation_conv_tracker.py.

import numpy
import argparse
import math
from scipy.linalg import expm
from scipy.interpolate import interp1d

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input_gt",
        help="Path to input file containing groundtruth bbox annotations",
    )
    parser.add_argument(
        "--input_rec_n",
        help="Path to input file containing estimation results on reconstructed frames",
    )
    parser.add_argument(
        "--input_img",
        help="Path to input file containing estimation results on conventional frames",
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with open(args.input_gt, "r") as file_gt:
        xx_gt_aux = []
        xy_gt_aux = []
        for line in file_gt:
            x, y, w, h = line.split(',')
            xmin, ymin, w, h = float(x), float(y), float(w), float(h)
            center_x = xmin + w / 2
            center_y = ymin + h / 2
            xx_gt_aux.append(center_x)
            xy_gt_aux.append(center_y) # Some labels might be missing. This will be adressed later

    with open(args.input_rec_n, "r") as file_rec:
        zx_rec = []
        zy_rec = []
        xx_rec = []
        xy_rec = []
        vx_rec = []
        vy_rec = []
        P_rec = []
        ts_rec_aux = []
        for line in file_rec:
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44, n = line.split(' ')
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44 = float(det), float(ts), float(zx), \
                    float(zy), float(xx), float(xy), float(vx), float(vy), float(P11), float(P12), float(P13), float(P14), \
                    float(P21), float(P22), float(P23), float(P24), float(P31), float(P32), float(P33), float(P34), float(P41), float(P42), float(P43), float(P44)
            if det == 1: # Only instants with available detection
                zx_rec.append(zx)
                zy_rec.append(zy)
                xx_rec.append(xx)
                xy_rec.append(xy)
                vx_rec.append(vx)
                vy_rec.append(vy)
                P = numpy.matrix([[P11, P12, P13, P14], [P21, P22, P23, P24], [P31, P32, P33, P34], [P41, P42, P43, P44]])
                P_rec.append(P)
                ts_rec_aux.append(ts)

    ts_gt_aux = []
    with open(args.input_img, "r") as file_img:
        zx_img = []
        zy_img = []
        xx_img = []
        xy_img = []
        vx_img = []
        vy_img = []
        P_img = []
        ts_img_aux = []
        for line in file_img:
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44, n = line.split(' ')
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44 = float(det), float(ts), float(zx), \
                        float(zy), float(xx), float(xy), float(vx), float(vy), float(P11), float(P12), float(P13), float(P14), \
                        float(P21), float(P22), float(P23), float(P24), float(P31), float(P32), float(P33), float(P34), float(P41), float(P42), float(P43), float(P44)
            ts_gt_aux.append(ts)
            if det == 1:
                zx_img.append(zx)
                zy_img.append(zy)
                xx_img.append(xx)
                xy_img.append(xy)
                vx_img.append(vx)
                vy_img.append(vy)
                P = numpy.matrix([[P11, P12, P13, P14], [P21, P22, P23, P24], [P31, P32, P33, P34], [P41, P42, P43, P44]])
                P_img.append(P)
                ts_img_aux.append(ts)


    # Timestamps starting from zero
    ts_rec = []
    for i in range(len(ts_rec_aux)):
        ts_rec.append(ts_rec_aux[i] - ts_rec_aux[0])
    tmax = ts_rec[-1]

    ts_img = []
    for i in range(len(ts_img_aux)):
        ts_img.append(ts_img_aux[i] - ts_rec_aux[0])

    ts_gt_all = []
    for i in range(len(ts_gt_aux)):
        ts_gt_all.append(ts_gt_aux[i] - ts_rec_aux[0])

    # Remove absent groundtruth labels
    xx_gt = []
    xy_gt = []
    ts_gt = []
    for i in range(len(ts_gt_all)):
        if ts_gt_all[i] < tmax:
            if xx_gt_aux[i] != 0 or xy_gt_aux[i] != 0:
                xx_gt.append(xx_gt_aux[i])
                xy_gt.append(xy_gt_aux[i])
                ts_gt.append(ts_gt_all[i])

    # Create values of z_gt for all t
    z_gt_x = interp1d(ts_gt, xx_gt)
    z_gt_y = interp1d(ts_gt, xy_gt)


    # Compute E_gt

    # System matrices
    A = numpy.matrix('0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0')
    B = numpy.matrix('0 0; 0 0; 1 0; 0 1')
    H = numpy.matrix('1 0 0 0; 0 1 0 0')
    Q0 = numpy.matrix('10 0; 0 10')

    # Integration
    int = 0
    time_rec = []
    pos_x_rec = []
    pos_y_rec = []
    # For each interval between measurements
    for i in range(len(ts_rec)-1):
        if ts_gt[0] < ts_rec[i] < tmax:
            time_between_measurements = ts_rec[i + 1] - ts_rec[i]
            nbins = 1000
            inc = time_between_measurements / nbins
            t = ts_rec[i]
            xv = [t]
            while t < ts_rec[i + 1]:
                t = t + inc
                if t > tmax:
                    xv.append(tmax)
                    break
                else:
                    xv.append(t)
            x_k = numpy.matrix([[xx_rec[i]], [xy_rec[i]], [vx_rec[i]], [vy_rec[i]]])
            result = []
            # Compute estimation for every t with no available measurement, relying on process knowledge
            for j in range(len(xv)):
                if xv[j] < ts_gt[-1]:
                    F = expm(A * (xv[j] - ts_rec[i]))
                    x = F * x_k
                    pos_x_rec.append(x[0,0])
                    pos_y_rec.append(x[1, 0])
                    time_rec.append(xv[j])
                    z_gt = numpy.matrix([[z_gt_x(xv[j])], [z_gt_y(xv[j])]])
                    err = pow(numpy.linalg.norm(z_gt - H * x), 2)
                    result.append(err)
            # Compute integral
            for k in range(len(result) - 1):
                int = int + (xv[k + 1] - xv[k]) * (result[k + 1] + result[k]) / 2

    E_gt_rec = math.sqrt((1 / (tmax - ts_rec[0])) * int)

    # On conventional frames
    int_img = 0
    time_img = []
    pos_x_img = []
    pos_y_img = []
    for i in range(len(ts_img) - 1):
        if ts_gt[0] < ts_img[i] < tmax:
            time_between_measurements = ts_img[i + 1] - ts_img[i]
            nbins = 1000
            inc = time_between_measurements / nbins
            t = ts_img[i]
            xv = [t]
            while t < ts_img[i + 1]:
                t = t + inc
                if t > tmax:
                    xv.append(tmax)
                    break
                else:
                    xv.append(t)
            x_k = numpy.matrix([[xx_img[i]], [xy_img[i]], [vx_img[i]], [vy_img[i]]])
            result = []
            for j in range(len(xv)):
                if xv[j] < ts_gt[-1]:
                    F = expm(A * (xv[j] - ts_img[i]))
                    x = F * x_k
                    pos_x_img.append(x[0,0])
                    time_img.append(xv[j])
                    pos_y_img.append(x[1,0])
                    z_gt = numpy.matrix([[z_gt_x(xv[j])], [z_gt_y(xv[j])]])
                    err = pow(numpy.linalg.norm(z_gt - H * x), 2)
                    result.append(err)
            for k in range(len(result) - 1):
                int_img = int_img + (xv[k + 1] - xv[k]) * (result[k + 1] + result[k]) / 2

    E_gt_img = math.sqrt((1 / (tmax - ts_img[0])) * int_img)


    # Compute E_x
    # On reconstructed frames
    int_P = 0
    stdev_x_rec = []
    stdev_y_rec = []
    for i in range(len(ts_rec)-1):
        if ts_gt[0] < ts_rec[i] < tmax:
            time_between_measurements = ts_rec[i + 1] - ts_rec[i]
            nbins = 1000
            inc = time_between_measurements / nbins
            t = ts_rec[i]
            xv = [t]
            while t < ts_rec[i + 1]:
                t = t + inc
                if t > tmax:
                    xv.append(tmax)
                    break
                else:
                    xv.append(t)
            P_k = P_rec[i]
            # Compute estimation for each time instant
            result = []
            for j in range(len(xv)):
                F = expm(A * (xv[j] - ts_rec[i]))
                G = numpy.linalg.inv(numpy.eye(4) - (xv[j] - ts_rec[i]) * A / 2) * B * (xv[j] - ts_rec[i])
                P = F * P_k * numpy.transpose(F) + G * Q0 * numpy.transpose(G)
                result.append(numpy.trace(P))
                if xv[j] < ts_gt[-1]:
                    stdev_x_rec.append(math.sqrt(P[0,0]))
                    stdev_y_rec.append(math.sqrt(P[1,1]))

            # Compute integral
            for k in range(len(xv) - 1):
                int_P = int_P + (xv[k + 1] - xv[k]) * (result[k + 1] + result[k]) / 2

    E_x_rec = math.sqrt((1 / (tmax - ts_rec[0])) * int_P)

    # On conventional frames
    int_P_img = 0
    stdev_x_img = []
    stdev_y_img = []
    Q0 = numpy.matrix('10 0; 0 10')
    for i in range(len(ts_img) - 1):
        if ts_gt[0] < ts_img[i] < tmax:
            time_between_measurements = ts_img[i + 1] - ts_img[i]
            nbins = 1000
            inc = time_between_measurements / nbins
            t = ts_img[i]
            xv = [t]
            while t < ts_img[i + 1]:
                t = t + inc
                if t > tmax:
                    xv.append(tmax)
                    break
                else:
                    xv.append(t)
            P_k = P_img[i]
            result = []
            for j in range(len(xv)):
                F = expm(A * (xv[j] - ts_img[i]))
                G = numpy.linalg.inv(numpy.eye(4) - (xv[j] - ts_img[i]) * A / 2) * B * (xv[j] - ts_img[i])
                P = F * P_k * numpy.transpose(F) + G * Q0 * numpy.transpose(G)
                result.append(numpy.trace(P))
                if xv[j] < ts_gt[-1]:
                    stdev_x_img.append(math.sqrt(P[0,0]))
                    stdev_y_img.append(math.sqrt(P[1,1]))
            for k in range(len(xv) - 1):
                int_P_img = int_P_img + (xv[k + 1] - xv[k]) * (result[k + 1] + result[k]) / 2

    E_x_img = math.sqrt((1/(tmax - ts_img[0])) * int_P_img)

    print('E_gt_rec: ', E_gt_rec)
    print('E_gt_img: ', E_gt_img)
    print('E_x_rec: ', E_x_rec)
    print('E_x_img: ', E_x_img)







