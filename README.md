# Synchronized RGB-D + IMU from OAK Camera for RTAB-Map

This repository provides a synchronized stereo vision pipeline using the **Luxonis OAK** camera. It generates **disparity-based RGB-D images** using
a custom stereo matching network along with **IMU data**, making it fully compatible with **RTAB-Map** for real-time 3D SLAM and map reconstruction.

---

##  Features

-  **Synchronized** stereo RGB and IMU data streams
-  Real-time **disparity estimation** with TensorRT acceleration
-  Generates **depth images** from stereo pairs using a deep learning model
-  Publishes calibrated and synchronized **IMU** data
-  ROS 2 compatible node
-  Fully compatible with **RTAB-Map** for 3D mapping

---

##  Applications

- Visual-Inertial SLAM
- Real-time 3D Reconstruction
- Robot Perception & Navigation
- Autonomous Mapping Systems

---

##  Requirements

- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html) 
- [DepthAI SDK](https://docs.luxonis.com/software/) 
- [TensorRT](https://developer.nvidia.com/tensorrt) + CUDA 
- [RTAB-Map](https://github.com/introlab/rtabmap)
- OAK Camera (OAK-D LR with RGB images tested)

Note 1: The code is written for left and right RGB images. The code can be adjusted for mono encoding by modifying a few lines in depthai_oakdpro_cuda_node.cpp.

Note 2: Modify depthai_desc package to get the exact camera description compatible with your OAK camera.

---

<p align="center" style="margin:0">
<img src="./imgs/odom_optimized.gif" alt="Path Following" width="600" border="0" />
</p>


##  Quick Start

Note: adjust the TensorRT paths in the CMakeLists.txt

```bash
mkdir -p depthai_rgbd_oak/src
cd depthai_rgbd_oak/src
git clone https://github.com/M2219/RGBD_OakCamera_RTABMap
cd ..
colcon build
source install/setup.bash
```

### Stereo Matching Network
Use ```onnx_transformed.py``` to transfer the stereo matching model to .onnx and 
use the below command to generate .plan file:

```bash
/usr/local/TensorRT-10.11.0.33/bin/trtexec --onnx=StereoModel.onnx --noTF32 --saveEngine=StereoModel.plan
```
Alternatively, you can use ESMStereo for real-time and accurate stereo matching model: Follow the instruction in this repo [ESMStereo](https://github.com/M2219/ESMStereo)

and 

```bash
cp StereoModel.plan /tmp
```
### Terminal 1: Oak camera publilsher

```bash
ros2 launch depthai_oakdpro depthai_oakdpro_cuda_node.launch.py
```

### Terminal 2: RTAB-map 

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  args:="--delete_db_on_start" \
  rgb_topic:=/left/image_rect \
  depth_topic:=/depth/image_raw \
  camera_info_topic:=/left/camera_info \
  imu_topic:=/oak/imu \
  frame_id:=oak_stereo_frame \
  approx_sync:=true \
  approx_sync_max_interval:=0.001 \
  wait_imu_to_init:=true
```

### Camera description

The depthai_desc package provides the camera's URDF and static transforms, 
currently configured for the Luxonis OAK-D Pro. It publishes the necessary 
/tf_static for frame relationships (e.g., oak_rgb_optical_frame, oak_left_camera_frame, etc.).

```bash
ros2 launch depthai_desc urdf_oak_launch.py 
```

### Usage with open3D

```bash
mkdir ~/open3d_data
```

1. Run the package and the RGBD data will be stored in ~/open3d_data/OakCamera

2. Adjust the configs based on your Oak camera model in ```config```

3. Build [open3D](https://github.com/isl-org/Open3D.git)

```bash
cd Open3D/examples/python/t_reconstruction_system
python3 dense_slam_gui.py --config /path/to/oak_config.yml
```
<p align="center" style="margin:0">
<img src="./imgs/open3d.gif" alt="Path Following" width="600" border="0" />
</p>



### Settings

The following parameters can be adjusted in the launch file

```bash
depthai_node = Node(
    package='depthai_oakdpro',
    executable='depthai_oakdpro_cuda_node',
    name='depthai_oakdpro_cuda_node',
    output='screen',
    parameters=[{
        'fx': 379.0, # focal legnth
        'baseline': 0.15, # stereo baseline
        'width': 640, # camera width
        'height': 400, # camera height
        'net_input_width': 640, # network input width
        'net_input_height': 384, # network input height
        'Imux': 0.0, # Imu x offset from the left camera
        'Imuy': -0.02, # Imu y offset from the left camera
        'Imuz': 0.0, # Imu z offset from the left camera
        'open3D_save': True # Save data to use in Open3D reconstruction
    }]
)
