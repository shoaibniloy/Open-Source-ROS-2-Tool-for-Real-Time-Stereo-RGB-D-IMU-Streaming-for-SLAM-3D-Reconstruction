# Synchronized RGB-D + IMU from OAK Camera for RTAB-Map

This repository offers a synchronized stereo vision pipeline using the **Luxonis OAK** camera. It produces **disparity-based RGB-D images** via a custom stereo matching network alongside **IMU data**, making it fully compatible with **RTAB-Map** for real-time SLAM and 3D map reconstruction.

---

## Features

- Provides **synchronized** stereo RGB and IMU data streams
- Real-time **disparity computation** accelerated with TensorRT
- Generates **depth images** from stereo pairs using a deep learning network
- Publishes calibrated, synchronized **IMU measurements**
- ROS 2 node integration
- Seamless compatibility with **RTAB-Map** for 3D mapping

---

## Applications

- Visual-inertial SLAM
- Online 3D reconstruction
- Robotic perception and navigation
- Autonomous mapping

---

## Requirements

- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html)
- [DepthAI SDK](https://docs.luxonis.com/software/)
- [TensorRT](https://developer.nvidia.com/tensorrt) + CUDA
- [RTAB-Map](https://github.com/introlab/rtabmap)
- OAK Camera (tested on OAK-D LR with RGB images)

**Note 1:** The implementation assumes left and right RGB images. For mono encoding, minor modifications are needed in `depthai_oakdpro_cuda_node.cpp`.

**Note 2:** Update the `depthai_desc` package to ensure camera description matches your OAK device.

---

<p align="center" style="margin:0">
<img src="./imgs/odom_optimized.gif" alt="Path Following" width="600" border="0" />
</p>

---

## Quick Start

Make sure to update TensorRT paths in the `CMakeLists.txt`:

```bash
mkdir -p depthai_rgbd_oak/src
cd depthai_rgbd_oak/src
git clone https://github.com/M2219/RGBD_OakCamera_RTABMap
cd ..
colcon build
source install/setup.bash
````

### Stereo Matching Network

Convert the stereo model to ONNX and generate a TensorRT plan file:

```bash
/usr/local/TensorRT-10.11.0.33/bin/trtexec --onnx=StereoModel.onnx --noTF32 --saveEngine=StereoModel.plan
```

Alternatively, use **ESMStereo** for accurate and real-time stereo matching: [ESMStereo](https://github.com/M2219/ESMStereo).

```bash
cp StereoModel.plan /tmp
```

### Terminal 1: Oak camera publisher

```bash
ros2 launch depthai_oakdpro depthai_oakdpro_cuda_node.launch.py
```

### Terminal 2: RTAB-Map

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

The `depthai_desc` package provides the URDF and static transforms for the Luxonis OAK-D Pro. It publishes the required `/tf_static` frames (e.g., `oak_rgb_optical_frame`, `oak_left_camera_frame`).

```bash
ros2 launch depthai_desc urdf_oak_launch.py 
```

---

### Usage with Open3D

```bash
mkdir ~/open3d_data
```

1. Run the package — RGB-D data will be saved to `~/open3d_data/OakCamera`.
2. Update the configs in `config` based on your OAK model.
3. Build [Open3D](https://github.com/isl-org/Open3D.git).

```bash
cd Open3D/examples/python/t_reconstruction_system
python3 dense_slam_gui.py --config /path/to/oak_config.yml
```

<p align="center" style="margin:0">
<img src="./imgs/open3d.gif" alt="Path Following" width="600" border="0" />
</p>

---

### Settings

The following parameters can be tuned in the launch file:

```python
depthai_node = Node(
    package='depthai_oakdpro',
    executable='depthai_oakdpro_cuda_node',
    name='depthai_oakdpro_cuda_node',
    output='screen',
    parameters=[{
        'fx': 379.0, # focal length
        'baseline': 0.15, # stereo baseline
        'width': 640, # image width
        'height': 400, # image height
        'net_input_width': 640, # network input width
        'net_input_height': 384, # network input height
        'Imux': 0.0, # IMU offset x from left camera
        'Imuy': -0.02, # IMU offset y from left camera
        'Imuz': 0.0, # IMU offset z from left camera
        'open3D_save': True # store data for Open3D reconstruction
    }]
)
