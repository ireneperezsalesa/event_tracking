# Estimation with our proposed framework.

import numpy
import sys
import os
import cv2
import argparse
import zipfile
import matplotlib.pyplot as plt
from scipy.linalg import expm
sys.path.insert(0, './e2vid/rpg_e2vid/')
from e2vid.rpg_e2vid.model.model import *
from e2vid.rpg_e2vid.utilsf.inference_utils import events_to_voxel_grid_pytorch
from e2vid.rpg_e2vid.image_reconstructor import ImageReconstructor
from e2vid.rpg_e2vid.options.inference_options import set_inference_options
from e2vid.rpg_e2vid.utilsf.loading_utils import load_model
from e2vid.rpg_e2vid.utilsf.timers import CudaTimer

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Path to txt or zip file containing events",
    )

    parser.add_argument('--num_events_per_pixel', default=0.35, type=float)
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)
    parser.add_argument('--det_class', type=int)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    set_inference_options(parser)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load E2VID model
    model = load_model('./e2vid/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar')
    model = model.to(device)
    model.eval()
    print('E2VID loaded')

    # Load YOLO model
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo = yolo.to(device)
    print('YOLOv5 loaded')

    # Covariance of the measurement noise - Value may need to be adapted
    #Rk = numpy.matrix('50 10; 10 50')
    if args.det_class == 0:
        Rk = numpy.matrix('2000 160; 160 1000')
    if args.det_class == 15:
        Rk = numpy.matrix('2000 1000; 1000 2000')

    # Matrices to describe the dynamic of the system
    A = numpy.matrix('0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0')
    B = numpy.matrix('0 0; 0 0; 1 0; 0 1')
    H = numpy.matrix('1 0 0 0; 0 1 0 0')
    H_t = numpy.transpose(H)
    Q = numpy.matrix('10 0; 0 10') # [px]    - Rk and Q parameters may be adjusted according to each sequence

    # Open event file
    if args.input.endswith('zip'):
        zip_file = zipfile.ZipFile(args.input)
        files_in_archive = zip_file.namelist()
        assert (len(files_in_archive) == 1)  # make sure there is only one text file in the archive
        event_file = zip_file.open(files_in_archive[0], 'r')
    else:
        event_file = open(args.input, 'r')

    # get width and height from text file header
    for line in event_file:
        if args.input.endswith('zip'):
            line = line.decode("utf-8")
        width, height, n = line.split(' ')
        width, height = int(width), int(height)
        break
    # ignore header + the first start_index lines
    start_index = 0 + args.skipevents
    for i in range(1 + start_index):
        event_file.readline()

    # Configure image reconstructor
    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)
    # Number of events in each reconstruction
    N = args.num_events_per_pixel * height * width
    print('N: ', N)

    # Set initial values
    x_k = numpy.matrix('0; 0; 0; 0')
    P_k = numpy.matrix('10000 0 0 0; 0 10000 0 0; 0 0 10000 0; 0 0 0 10000')
    k = 0
    p = 0
    sim_time = 0
    total_cost = 0

    states = None
    last_stamp = None
    z_k = numpy.matrix('0; 0')  
    x_k_accum = x_k
    z_k_accum = x_k[0:2,:]
    time_stamp_list = []
    P = P_k.ravel().tolist()[0]
    P_list = [P]
    available_det = [0]

    # Total time
    total_time = 20 # Max time of the sequence that you want to reconstruct

    first_det_available = False

    # Algorithm for estimation
    while sim_time < total_time:

        print('image: ', k)

        # MEASUREMENT ------------------------------------------------------------------------------------------------------

        # Get events in event window of size N
        event_list = []
        i = 0
        for line in event_file:
            if args.input.endswith('zip'):
                line = line.decode("utf-8")
            t, x, y, pol = line.split(' ')
            t, x, y, pol = float(t), int(x), int(y), int(pol)
            stamp = t
            i = i + 1
            if i > N:
                break
            event_list.append([t, x, y, pol])
            if last_stamp is None:
                last_stamp = t # initialize timestamp to first event of the sequence
                start_ts = t
                time_stamp_list.append(t)

        event_window = numpy.array(event_list)

        # Turn events into torch tensor
        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                    num_bins=5,
                                                    width=width,
                                                    height=height,
                                                    device=device)

        # Time between last available measurement and current moment
        dt = stamp - last_stamp

        # Compute Ad and Qd according to time dt
        F = expm(A * dt)
        G = numpy.linalg.inv(numpy.eye(4) - dt * A / 2) * B * dt
        Qd = G * Q * numpy.transpose(G)

        # DATA PROCESSING ---------------------------------------------------------------------------------------------------

        # Reconstruct frame and apply detection network to produce a measurement
        with CudaTimer('RecDet'):
            num_events_in_window = event_window.shape[0]
            event_tensor_id = start_index + num_events_in_window
            with CudaTimer('Rec'):
                reconstructor.update_reconstruction(event_tensor, event_tensor_id, last_stamp)
            start_index += num_events_in_window

            # Detect person
            frame_path = os.path.join(args.output_folder, 'reconstruction/frame_{:010d}.png'.format(event_tensor_id))
            img = cv2.imread(frame_path)[..., ::-1]
            with CudaTimer('Det'):
                result = yolo(img, size=width)

            # Compute measurement (image coordinates for the center of the bounding box)
            predictions = result.xyxy[0]
            n, m = predictions.shape
            min_confidence = 0
            det_present = False
            for i in range(n):
                if predictions[i, m - 1] == args.det_class:  # class 0 == 'person'
                    if first_det_available is False:
                        first_det_available = True
                    zx = (predictions[i, 0] + predictions[i, 2]) / 2
                    zy = (predictions[i, 1] + predictions[i, 3]) / 2
                    det_present = True
                    if predictions[
                        i, m - 2] > min_confidence:  # if there is more than one detection, pick the one with highest confidence
                        z_k[0, 0] = zx.item()
                        z_k[1, 0] = zy.item()
                        min_confidence = predictions[i, m - 2]

            # Uncomment these lines if you wish to save detection results on each frame
            #if not os.path.isdir(os.path.join(args.output_folder, 'detect/')):
            #    os.mkdir(os.path.join(args.output_folder, 'detect/'))
            #det_path = os.path.join(args.output_folder, 'detect/frame_{:010d}'.format(event_tensor_id))
            #os.mkdir(det_path)
            #result.save(det_path)


        # ESTIMATION --------------------------------------------------------------------------------------------------------

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

        sim_time = stamp - start_ts
        time_stamp_list.append(stamp)

        k = k + 1


        # PLOT Z_K AND X_K OVER THE IMAGE
        implot = plt.imshow(img)
        scat_x = plt.scatter(x, y, s=10, c='r', marker="s", label='x_img')
        scat_z = plt.scatter(zx, zy, s=10, c='b', marker="o", label='z_img')
        plt.axis('off')
        if not os.path.isdir(os.path.join(args.output_folder, 'estim_traj/')):
            os.mkdir(os.path.join(args.output_folder, 'estim_traj/'))
        plot_path = os.path.join(args.output_folder, 'estim_traj/frame_{:010d}.png'.format(event_tensor_id))
        plt.savefig(plot_path)
        scat_x.remove()
        scat_z.remove()

    # End of loop


    # TRANSFER ESTIMATION RESULTS TO TXT FILE
    data = numpy.concatenate((numpy.transpose(numpy.asmatrix(available_det)), numpy.transpose(numpy.asmatrix(time_stamp_list)), numpy.transpose(z_k_accum), numpy.transpose(x_k_accum), numpy.asmatrix(P_list)), axis=1)
    with open(os.path.join(args.output_folder, 'estim_event.txt'), 'w') as f:
        for line in data:
            string = ''
            for i in range(24): # write available_det (1/0 for yes/no), timestamp, zk_x, zk_y, xk_x, xk_y, xk_vx, xk_vy, P
                string = string + str(line[0,i]) + ' '
            string = string + '\n'
            f.write(string)



