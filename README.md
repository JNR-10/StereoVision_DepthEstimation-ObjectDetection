# Depth Estimation From Stereo Images

## Introduction:
assets/demo_output.mp4

(Note: Upper part is Disparity Map and bottom part is Object detection + Depth Estimation)

Incase of Stereo Setup, Depth estimation is dependent on disparity map.
![disparity drawio](https://user-images.githubusercontent.com/22910010/221393481-38847a4e-3c24-4daf-a803-e948051be575.png)

[PointCloud Output]

assets/point_cloud.mp4

## Dependency

- Download Pre-Trained model which i shared at [Download Link](https://drive.google.com/drive/folders/1j9DHSvIqM41vMVLlJomt8WRGRgBa_5h1?usp=share_link)

    Place it inside root folder and update the path in the config.py.

    ```
    RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
    FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
    ...
    ```

- Download Yolo for object detection.I shared it at [Download Link](https://drive.google.com/file/d/1onQwWb4lrJ4a-OLLHcgwM0YD6-H9U6rh/view?usp=share_link).

## Setting up DataSet
Download Kitti Dataset from [Download Link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

 - Download Left/Right Images: Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)
 - Download Calibration files: Download calibration files (1 MB)

Keep these files in some path and update config.py


```
[config.py]
KITTI_CALIB_FILES_PATH=".../kitti_stereo_2015/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_3/*.png"
...
```

## How to use


Run "python3 demo.py" change the configuration in config.py in order to run different architecture such as BGNet, CreStereo, RAFT-Stereo etc.

```
KITTI_CALIB_FILES_PATH=".../kitti_stereo_2015/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH=".../kitti_stereo_2015/testing/image_3/*.png"

RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
DEVICE = "cuda"

# raft-stereo=0, fastacv-plus=1, bgnet=2, gwcnet=3, pasmnet=4, crestereo=5, hitnet=6, psmnet=7
ARCHITECTURE_LIST = ["raft-stereo", "fastacv-plus", "bgnet", 'gwcnet', 'pasmnet', 'crestereo', 'hitnet', 'psmnet']
ARCHITECTURE = ARCHITECTURE_LIST[1]
SAVE_POINT_CLOUD = 0
SHOW_DISPARITY_OUTPUT = 1
SHOW_3D_PROJECTION = 0
```

## Evaluation
Different state of the art (SOTA) deep learning based architetures are proposed to solve disparity and are given below:

![disparity_timeline drawio(1)](https://user-images.githubusercontent.com/22910010/221393628-17f66ca6-7255-45a4-8faf-46d768075a32.png)

Here is the profiling data:

![disparity_map_profile_](https://user-images.githubusercontent.com/22910010/221400837-5ad3ae24-f23f-420a-9b4d-8328c1499c21.png)

Here is the inference time on Nvidia-2080Ti

![inference drawio](https://user-images.githubusercontent.com/22910010/221400886-c5ed6e1b-1e7e-4bcd-b6d9-5709f4503863.png)

# Acknowledgements
  Thanks to the authors of fastacv-plus, bgnet, gwcnet, pasmnet, crestereo, hitnet, psmnet and raft-stereo for their opensource code.
 
# References
- https://github.com/princeton-vl/RAFT-Stereo.git.
- https://github.com/gangweiX/Fast-ACVNet.
- https://github.com/3DCVdeveloper/BGNet.
- https://github.com/megvii-research/CREStereo.
- https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation.
- https://github.com/xy-guo/GwcNet.
- https://github.com/JiaRenChang/PSMNet.
- https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM/tree/master/PASMnet.

---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)






