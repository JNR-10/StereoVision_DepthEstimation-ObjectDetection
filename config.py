KITTI_CALIB_FILES_PATH="/Users/spartan/Downloads/StereoVision_DepthEstimation-ObjectDetection/data_scene_flow_calib/testing/calib_cam_to_cam/*.txt"
KITTI_LEFT_IMAGES_PATH="/Users/spartan/Downloads/StereoVision_DepthEstimation-ObjectDetection/data_scene_flow/testing/image_2/*.png"
KITTI_RIGHT_IMAGES_PATH="/Users/spartan/Downloads/StereoVision_DepthEstimation-ObjectDetection/data_scene_flow/testing/image_3/*.png"

RAFT_STEREO_MODEL_PATH = "pretrained_models/raft_stereo/raft-stereo_20000.pth"
FASTACV_MODEL_PATH = "pretrained_models/fast_acvnet/kitti_2015.ckpt"
BGNET_PLUS_MODEL_PATH = "pretrained_models/bgnet/kitti_15_BGNet_Plus.pth"
GWCNET_MODEL_PATH = "pretrained_models/gwcnet/gwcnet_g.ckpt"
PASMNET_MODEL_PATH = "pretrained_models/pasmnet/PASMnet_192_kitti_epoch5101.pth.tar"

CRESTEREO_MODEL_PATH = "pretrained_models/crestereo/crestereo_eth3d.pth"
HITNET_MODEL_PATH = "pretrained_models/hitnet/bestD1_checkpoint.ckpt"
PSMNET_MODEL_PATH = 'pretrained_models/psmnet/pretrained_model_KITTI2015.tar'
DEVICE = "cuda:0"
DEVICE1 = "cuda:1"


# raft-stereo=0, fastacv-plus=1, bgnet=2, gwcnet=3, pasmnet=4, crestereo=5, hitnet=6, psmnet=7

ARCHITECTURE_LIST = ["raft-stereo", "fastacv-plus", "bgnet", 'gwcnet', 'pasmnet', 'crestereo', 'hitnet', 'psmnet']
ARCHITECTURE = ARCHITECTURE_LIST[2]
SAVE_POINT_CLOUD = 1
POINT_CLOUD_PATH = "output/point_clouds/point_cloud.ply"
OUTPUT_PATH = "output/disparity_maps/"
SHOW_DISPARITY_OUTPUT = 1
SHOW_3D_PROJECTION = 0

# Enable this Profile flag only when you want to Profile
PROFILE_FLAG = 0
PROFILE_IMAGE_WIDTH = 512
PROFILE_IMAGE_HEIGHT = 512

# Video output settings for demo.py
VIDEO_SAVE = 1
VIDEO_PATH = "output/demo_output.mp4"
VIDEO_FPS = 3
