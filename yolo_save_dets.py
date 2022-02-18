# Create files containing all detected bounding boxes on a sequence. Can be executed on conventional frames or
# intensity frames reconstructed from events.

import torch
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--input",
        help="Path to input directory containing images",
    )
    parser.add_argument(
        "--output",
        help="Path to output txt file"
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        help="Image resolution: width height"
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    w, h = args.img_size[0], args.img_size[1]

    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Images
    imgs_list = sorted(os.listdir(args.input))  # batch of images

    img_number = 0

    with open(args.output, "w") as file:
        for img in imgs_list:
            if img.endswith('png') or img.endswith('jpeg') or img.endswith('jpg') or img.endswith('bmp'):
                img = args.input + img
                results = model(img, size=max(w, h))
                # Results
                n, m = results.xyxy[0].size()
                for i in range(n):
                    xmin, ymin, xmax, ymax, conf, clase = results.xyxy[0][i, :]
                    string = str(img_number) + ' ' + str(xmin.item()) + ' ' + str(ymin.item()) + ' ' + str(
                        xmax.item()) + ' ' + str(ymax.item()) + ' ' + str(conf.item()) + ' ' + str(
                        int(clase.item())) + '\n'
                    file.write(string)
            img_number = img_number + 1




