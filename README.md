event_tracking
==============

Visual tracking with event cameras by using E2VID and YOLO networks. 

User requirements
-----------------

Our code has been tested on Ubuntu 16.04, in an Anaconda3 environment with Python 3.8 and the following configuration: 

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pandas
conda install scipy
pip install opencv-python
```

To run the event tracker, the E2VID repository needs to be cloned, along with their pretrained model. This can be found [here](https://github.com/uzh-rpg/rpg_e2vid).

A pretrained model of the YOLOv5 network is used by loading from the Pytorch Hub. See details [here](https://pytorch.org/hub/ultralytics_yolov5/).


Run the tracker
---------------

```bash
python estimation_event_tracker.py \
--input e2vid/rpg_e2vid/data/events.txt \
--output_folder output/ \
--det_class 0  \
--num_events_per_pixel 0.5
```

References
---------------

This work uses the E2VID and YOLOv5 neural networks.
 - H. Rebecq, R. Ranftl, V. Koltun, and D. Scaramuzza, “High speed andhigh dynamic range video with an event camera,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 6, pp. 1964–1980, 2021.
 - G. Jocher et al., “YOLOv5n ’Nano’ models, Roboflow integration, Tensor-Flow  export,  OpenCV  DNN  support,” doi:10.5281/zenodo.5563715,2021.
