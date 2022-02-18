# Estimation with baseline tracker using the conventional frames provided by the dataset.

import numpy
import torch
import os
import cv2
import argparse
import zipfile
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Path to folder containing frames",
    )
    parser.add_argument(
        "--input_ev",
        metavar="FILE",
        help="Path to txt or zip event file",
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder to save results"
    )
    parser.add_argument(
        "--det_class", type = int,
        help="Object class to be detected. "
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo = yolo.to(device)
    print('YOLOv5 loaded')

    # Covariance of the measurement noise - Value may need to be adapted
    if args.det_class == 0:
        Rk = numpy.matrix('1200 120; 120 1000')
    if args.det_class == 15:
        Rk = numpy.matrix('506.7972 -156.0028; -156.0028 344.1236')

    # Matrices to describe the dynamic of the system
    A = numpy.matrix('0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0')
    B = numpy.matrix('0 0; 0 0; 1 0; 0 1')
    H = numpy.matrix('1 0 0 0; 0 1 0 0')
    H_t = numpy.transpose(H)
    Q = numpy.matrix('10 0 ; 0 10')  # [px]    - Rk and Q parameters may be adjusted according to each sequence

    # Open associated event file
    if args.input_ev.endswith('zip'):
        zip_file = zipfile.ZipFile(args.input_ev)
        files_in_archive = zip_file.namelist()
        assert (len(files_in_archive) == 1)  # make sure there is only one text file in the archive
        event_file = zip_file.open(files_in_archive[0], 'r')
    else:
        event_file = open(args.input_ev, 'r')
    # get width and height from text file header
    for line in event_file:
        if args.input_ev.endswith('zip'):
            line = line.decode("utf-8")
        width, height, n = line.split(' ')
        width, height = int(width), int(height)
        break
    # ignore header + the first start_index lines
    start_index = 0
    for i in range(1 + start_index):
        event_file.readline()
    # get timestamps to compute dt
    first_ts = None
    last_ts = None
    for line in event_file:
        if args.input_ev.endswith('zip'):
            line = line.decode("utf-8")
        t, x, y, pol = line.split(' ')
        t, x, y, pol = float(t), int(x), int(y), int(pol)
        last_ts = t
        if first_ts is None:
            first_ts = t
            stamp = t
            last_stamp = t

    # Compute time increments between frames
    num_frames = len(os.listdir(args.input))
    time_between_frames = (last_ts - first_ts) / num_frames

    # Set initial values
    x_k = numpy.matrix('0; 0; 0; 0')
    P_k = numpy.matrix('10000 0 0 0; 0 10000 0 0; 0 0 10000 0; 0 0 0 10000')
    k = 0

    z_k = x_k[0:2,:]
    x_k_accum = x_k
    z_k_accum = x_k[0:2,:]
    time_stamp_list = [last_stamp]
    P = P_k.ravel().tolist()[0]
    P_list = [P]
    available_det = [0]

    first_det_available = False

    print('stamp: ', stamp)

    for frame in sorted(os.listdir(args.input)):
        print('img: ', k)

        stamp = stamp + time_between_frames

        # Compute discrete matrices according to time elapsed since last available measurement
        dt = stamp - last_stamp
        F = expm(A * dt)
        G = numpy.linalg.inv(numpy.eye(4) - dt * A / 2) * B * dt
        Qd = G * Q * numpy.transpose(G)

        # Get frame
        frame_path = os.path.join(args.input, frame)
        img = cv2.imread(frame_path)[..., ::-1]

        # Detection with YOLO
        result = yolo(img, size=width)

        # Choose most confident detection and compute z_k
        predictions = result.xyxy[0]
        n, m = predictions.shape
        min_confidence = 0
        det_present = False
        for i in range(n):
            if predictions[i, m - 1] == args.det_class:  # 0 person, 2 car, 15 cat
                if first_det_available is False:
                    first_det_available = True
                zx = (predictions[i, 0] + predictions[i, 2]) / 2
                zy = (predictions[i, 1] + predictions[i, 3]) / 2
                det_present = True
                if predictions[i, m - 2] > min_confidence:  # if there is more than one detection, pick the one with better confidence
                    z_k[0, 0] = zx.item()
                    z_k[1, 0] = zy.item()
                    min_confidence = predictions[i, m - 2]

        # Uncomment these lines if you wish to save detection results on each frame
        #if not os.path.isdir(os.path.join(args.output_folder, 'detect/')):
        #    os.mkdir(os.path.join(args.output_folder, 'detect/'))
        #det_path = os.path.join(args.output_folder, 'detect/', frame)
        #os.mkdir(det_path)
        #result.save(det_path)


        if first_det_available is True:
            # Classic Kalman: predict x_k using x_k1, estimate with prediction and measurement z_k
            # Predict
            x_k_k1 = F * x_k
            P_k_k1 = F * P_k * numpy.transpose(F) + Qd

            # Correct if measurement is available
            if det_present is True:
                L_k = P_k_k1 * H_t * numpy.linalg.inv(H * P_k_k1 * H_t + Rk)
                x_k = x_k_k1 + L_k * (z_k - H * x_k_k1)
                P_k = (numpy.eye(4) - L_k * H) * P_k_k1
                z_k_accum = numpy.concatenate((z_k_accum, z_k), axis=1)  # every column corresponds to an iteration
                x_k_accum = numpy.concatenate((x_k_accum, x_k), axis=1)
                P_unravel = P_k.ravel().tolist()[0]
                P_list.append(P_unravel)
                last_stamp = stamp # last stamp keeps the timestamp of the last available measurement
                available_det.append(1)
                x = x_k[0].item()
                y = x_k[1].item()
                zx = numpy.squeeze(numpy.asarray(z_k[0, :]))
                zy = numpy.squeeze(numpy.asarray(z_k[1, :]))
            else:
                z_k_k1 = H*x_k_k1
                z_k_accum = numpy.concatenate((z_k_accum, H*x_k_k1), axis=1)
                x_k_accum = numpy.concatenate((x_k_accum, x_k_k1), axis=1)
                P_unravel = P_k_k1.ravel().tolist()[0]
                P_list.append(P_unravel)
                available_det.append(0)
                x = x_k_k1[0].item()
                y = x_k_k1[1].item()
                zx = numpy.squeeze(numpy.asarray(z_k_k1[0, :]))
                zy = numpy.squeeze(numpy.asarray(z_k_k1[1, :]))

        else:
            # Keep everything at initial value until first detection
            # This part of the sequence will not be taken into account when computing error metrics
            z_k_accum = numpy.concatenate((z_k_accum, z_k), axis=1)  # every column corresponds to an iteration
            x_k_accum = numpy.concatenate((x_k_accum, x_k), axis=1)
            P_unravel = P_k.ravel().tolist()[0]
            P_list.append(P_unravel)
            last_stamp = stamp
            available_det.append(0)
            x = x_k[0].item()
            y = x_k[1].item()
            zx = numpy.squeeze(numpy.asarray(z_k[0, :]))
            zy = numpy.squeeze(numpy.asarray(z_k[1, :]))

        time_stamp_list.append(stamp)
        k = k + 1

        # PLOT Z_K AND X_K OVER THE IMAGE
        implot = plt.imshow(img)
        scat_x = plt.scatter(x, y, s=10, c='r', marker="s", label='x_img')
        scat_z = plt.scatter(zx, zy, s=10, c='b', marker="o", label='z_img')
        plt.axis('off')
        if not os.path.isdir(os.path.join(args.output_folder, 'estim_traj/')):
            os.mkdir(os.path.join(args.output_folder, 'estim_traj/'))
        plot_path = os.path.join(args.output_folder, 'estim_traj/', frame)
        plt.savefig(plot_path)
        scat_x.remove()
        scat_z.remove()

    # End of loop

    # TRANSFER DATA TO TXT FILE
    data = numpy.concatenate((numpy.transpose(numpy.asmatrix(available_det)), numpy.transpose(numpy.asmatrix(time_stamp_list)), numpy.transpose(z_k_accum), numpy.transpose(x_k_accum), numpy.asmatrix(P_list)), axis=1)
    with open(os.path.join(args.output_folder, 'estim_baseline.txt'), 'w') as f:
        for line in data:
            string = ''
            for i in range(24): # write available_det (1/0 for yes/no), timestamp, zk_x, zk_y, xk_x, xk_y, xk_vx, xk_vy, P
                string = string + str(line[0,i]) + ' '
            string = string + '\n'
            f.write(string)
