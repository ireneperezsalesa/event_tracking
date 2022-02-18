# Plot position and velocity estimations for baseline tracker and our proposal.
# Input: txt files produced by estimation_event_tracker.py and estimation_conv_tracker.py.

import numpy
import math
import argparse
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input_gt",
        help="Path to input file containing groundtruth bbox annotations",
    )
    parser.add_argument(
        "--input_rec_n",
        help="Path to input file containing estimation values with our tracker",
    )
    parser.add_argument(
        "--input_img",
        help="Path to input file containing estimation values with baseline tracker",
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
            xy_gt_aux.append(center_y)

    with open(args.input_rec_n, "r") as file_rec:
        zx_rec = []
        zy_rec = []
        xx_rec = []
        xy_rec = []
        vx_rec = []
        vy_rec = []
        xx_stdev_rec = []
        xy_stdev_rec = []
        vx_stdev_rec = []
        vy_stdev_rec = []
        ts_rec_aux = []
        for line in file_rec:
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44, n = line.split(' ')
            ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44 = float(ts), float(zx), \
                float(zy), float(xx), float(xy), float(vx), float(vy), float(P11), float(P12), float(P13), float(P14), \
                float(P21), float(P22), float(P23), float(P24), float(P31), float(P32), float(P33), float(P34), float(P41), float(P42), float(P43), float(P44)
            zx_rec.append(zx)
            zy_rec.append(zy)
            xx_rec.append(xx)
            xy_rec.append(xy)
            vx_rec.append(vx)
            vy_rec.append(vy)
            xx_stdev_rec.append(math.sqrt(P11))
            xy_stdev_rec.append(math.sqrt(P22))
            vx_stdev_rec.append(math.sqrt(P33))
            vy_stdev_rec.append(math.sqrt(P44))
            ts_rec_aux.append(ts)

    with open(args.input_img, "r") as file_img:
        zx_img = []
        zy_img = []
        xx_img = []
        xy_img = []
        vx_img = []
        vy_img = []
        xx_stdev_img = []
        xy_stdev_img = []
        vx_stdev_img = []
        vy_stdev_img = []
        ts_img_aux = []
        for line in file_img:
            det, ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44, n = line.split(' ')
            ts, zx, zy, xx, xy, vx, vy, P11, P12, P13, P14, P21, P22, P23, P24, P31, P32, P33, P34, P41, P42, P43, P44 = float(ts), float(zx), \
                float(zy), float(xx), float(xy), float(vx), float(vy), float(P11), float(P12), float(P13), float(P14), \
                float(P21), float(P22), float(P23), float(P24), float(P31), float(P32), float(P33), float(P34), float(P41), float(P42), float(P43), float(P44)
            zx_img.append(zx)
            zy_img.append(zy)
            xx_img.append(xx)
            xy_img.append(xy)
            vx_img.append(vx)
            vy_img.append(vy)
            xx_stdev_img.append(math.sqrt(P11))
            xy_stdev_img.append(math.sqrt(P22))
            vx_stdev_img.append(math.sqrt(P33))
            vy_stdev_img.append(math.sqrt(P44))
            ts_img_aux.append(ts)

    # Timestamp values starting from zero
    ts_rec = []
    for i in range(len(ts_rec_aux)):
        ts_rec.append(ts_rec_aux[i] - ts_rec_aux[0])

    ts_img = []
    k = 0
    for i in range(len(ts_img_aux)):
        ts_img.append(ts_img_aux[i] - ts_img_aux[0])

    ts_gt_aux = ts_img[1:]

    # Remove absent groundtruth labels
    xx_gt = []
    xy_gt = []
    ts_gt = []
    for i in range(len(ts_gt_aux)):
        if xx_gt_aux[i] != 0 or xy_gt_aux[i] != 0:
            xx_gt.append(xx_gt_aux[i])
            xy_gt.append(xy_gt_aux[i])
            ts_gt.append(ts_gt_aux[i])


    # Find first nonzero estimation
    skip_rec = 0
    for i in range(len(ts_rec)):
        if xx_rec[i] == 0.0:
            skip_rec = i

    skip_img = 0
    for i in range(len(ts_img)):
        if xx_img[i] == 0.0:
            skip_img = i

    # POSITION ESTIMATION PLOTS
    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.plot(numpy.asarray(ts_gt) - ts_rec[skip_rec], xx_gt, color='g', label='Groundtruth')
    plt.plot(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], xx_img[skip_img:], color='r', label='Baseline tracker')
    plt.fill_between(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], numpy.asarray(xx_img[skip_img:]) - 3*numpy.asarray(xx_stdev_img[skip_img:]), numpy.asarray(xx_img[skip_img:]) + 3*numpy.asarray(xx_stdev_img[skip_img:]), color='r', alpha=0.25, linewidth=0)
    plt.plot(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], xx_rec[skip_rec:], color='b', label='Our tracker')
    plt.fill_between(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], numpy.asarray(xx_rec[skip_rec:]) - 3*numpy.asarray(xx_stdev_rec[skip_rec:]), numpy.asarray(xx_rec[skip_rec:]) + 3*numpy.asarray(xx_stdev_rec[skip_rec:]), color='b', alpha=0.25, linewidth=0)
    plt.xlabel('Timestamps (s)')
    plt.ylabel('X-coordinate (px)')
    plt.legend(loc='lower right')
    plt.xlim([0, ts_rec[-1] - ts_rec[skip_rec]])
    #plt.ylim([-700, 700])
    plt.show()
    #plt.savefig('estim_pos_x.pdf', dpi=300, transparent=False, bbox_inches='tight')

    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.plot(numpy.asarray(ts_gt) - ts_rec[skip_rec], xy_gt, color='g', label='Groundtruth')
    plt.plot(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], xy_img[skip_img:], color='r', label='Baseline tracker')
    plt.fill_between(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], numpy.asarray(xy_img[skip_img:]) - 3*numpy.asarray(xy_stdev_img[skip_img:]), numpy.asarray(xy_img[skip_img:]) + 3*numpy.asarray(xy_stdev_img[skip_img:]), color='r', alpha=0.25, linewidth=0)
    plt.plot(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], xy_rec[skip_rec:], color='b', label='Our tracker')
    plt.fill_between(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], numpy.asarray(xy_rec[skip_rec:]) - 3*numpy.asarray(xy_stdev_rec[skip_rec:]), numpy.asarray(xy_rec[skip_rec:]) + 3*numpy.asarray(xy_stdev_rec[skip_rec:]), color='b', alpha=0.25, linewidth=0)
    plt.xlabel('Timestamps (s)')
    plt.ylabel('Y-coordinate (px)')
    plt.legend(loc='upper right')
    plt.xlim([0, ts_rec[-1] - ts_rec[skip_rec]])
    #plt.ylim([-500, 800])
    plt.show()
    #plt.savefig('estim_pos_y.pdf', dpi=300, transparent=False, bbox_inches='tight')



    # VELOCITY ESTIMATION PLOTS
    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.plot(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], vx_img[skip_img:], color='r', label='Baseline tracker')
    plt.fill_between(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec],
                     numpy.asarray(vx_img[skip_img:]) - 2 * numpy.asarray(vx_stdev_img[skip_img:]),
                     numpy.asarray(vx_img[skip_img:]) + 2 * numpy.asarray(vx_stdev_img[skip_img:]), color='r',
                     alpha=0.25, linewidth=0)
    plt.plot(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], vx_rec[skip_rec:], color='b', label='Our tracker')
    plt.fill_between(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec],
                     numpy.asarray(vx_rec[skip_rec:]) - 2 * numpy.asarray(vx_stdev_rec[skip_rec:]),
                     numpy.asarray(vx_rec[skip_rec:]) + 2 * numpy.asarray(vx_stdev_rec[skip_rec:]), color='b',
                     alpha=0.25, linewidth=0)
    plt.xlabel('Timestamps (s)')
    plt.ylabel('X-coordinate velocity (px/s)')
    plt.legend(loc='lower right')
    plt.xlim([0, ts_rec[-1] - ts_rec[skip_rec]])
    #plt.show()
    #plt.savefig('estim_vel_x.pdf', dpi=300, transparent=False, bbox_inches='tight')

    plt.style.use(['science', 'ieee'])
    plt.figure()
    plt.plot(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec], vy_img[skip_img:], color='r', label='Baseline tracker')
    plt.fill_between(numpy.asarray(ts_img[skip_img:]) - ts_rec[skip_rec],
                     numpy.asarray(vy_img[skip_img:]) - 2 * numpy.asarray(vy_stdev_img[skip_img:]),
                     numpy.asarray(vy_img[skip_img:]) + 2 * numpy.asarray(vy_stdev_img[skip_img:]), color='r',
                     alpha=0.25, linewidth=0)
    plt.plot(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec], vy_rec[skip_rec:], color='b', label='Our tracker')
    plt.fill_between(numpy.asarray(ts_rec[skip_rec:]) - ts_rec[skip_rec],
                     numpy.asarray(vy_rec[skip_rec:]) - 2 * numpy.asarray(vy_stdev_rec[skip_rec:]),
                     numpy.asarray(vy_rec[skip_rec:]) + 2 * numpy.asarray(vy_stdev_rec[skip_rec:]), color='b',
                     alpha=0.25, linewidth=0)
    plt.xlabel('Timestamps (s)')
    plt.ylabel('Y-coordinate velocity (px/s)')
    plt.legend(loc='lower right')
    plt.xlim([0, ts_rec[-1] - ts_rec[skip_rec]])
    #plt.show()
    #plt.savefig('estim_vel_y.pdf', dpi=300, transparent=False, bbox_inches='tight')