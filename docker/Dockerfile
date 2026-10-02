# ROS 2 Humble with RViz and Gazebo included (matches the project's Humble requirement)
FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Tools for ALL phases of the project, installed once so we never fight installs later:
#   robot_localization -> EKF (Phase 2)      navigation2 / nav2_bringup -> Nav2 (Phase 4)
#   gazebo_ros_pkgs    -> Gazebo simulation  xacro / robot_state_publisher -> URDF (Phase 2)
#   stereo_image_proc / image_proc / pcl_ros -> stereo depth and point clouds (Phase 3)
#   vision_msgs / slam_toolbox / torch + ultralytics / gtsam -> object detection and SLAM (Phase 3, Task 7)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-serial \
        python3-colcon-common-extensions \
        python3-rosdep \
        git nano sudo \
        ros-humble-robot-localization \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher-gui \
        ros-humble-tf2-tools \
        ros-humble-stereo-image-proc \
        ros-humble-image-proc \
        ros-humble-pcl-ros \
        ros-humble-pcl-conversions \
        python3-numpy \
        python3-pip \
        ros-humble-vision-msgs \
        ros-humble-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

# YOLO object detection (Phase 3, Task 7): PyTorch for the CPU, then ultralytics (YOLOv8) and the
# text encoder CLIP (package clip-anytorch) that YOLOv8-World uses once to turn words like
# "pallet" into a detector. (The git version of CLIP failed to install silently, so we use PyPI.)
# numpy stays below 2 because the ROS packages in this image were built against numpy 1.x.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir "numpy<2" ultralytics clip-anytorch gtsam==4.2

# A normal (non-root) user whose ID matches your Ubuntu user (usually 1000),
# so files created inside Docker belong to you, not to root.
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ros \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ros \
    && echo "ros ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ros

# Every new terminal inside the container: turn on ROS 2 and (if built) our workspace
RUN echo 'source /opt/ros/humble/setup.bash' >> /home/ros/.bashrc \
    && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /home/ros/.bashrc

USER ros
WORKDIR /ws
CMD ["bash"]
