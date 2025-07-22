FROM osrf/ros:jazzy-desktop-full-noble

# ???
RUN DEBIAN_FRONTEND=noninteractive
#ENV DEBIAN_FRONTEND=noninteractive

# Libs and dependencies : git, ROS2, rviz2, gazebo, PCL, terminator, cmake, colcon, eigen ...
RUN apt-get update && apt upgrade -y && apt-get install -y \
    terminator \
    tmux -y \    
    git-all -y \
    lsb-release -y \
    tmux -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev  \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libabsl-dev \
    libpcl-dev \
    wget \
    x11-apps -y \
    ros-jazzy-rviz2 \
    ros-jazzy-ros-gz \
    ros-jazzy-cv-bridge \
    python3-colcon-common-extensions \
    ros-jazzy-tf-transformations \	
    && rm -rf /var/lib/apt/lists/* 


# OPENCV + CONTRIB    
RUN     git clone https://github.com/opencv/opencv.git
RUN	git clone https://github.com/opencv/opencv_contrib.git 
RUN     cd opencv \
        &&  mkdir release \
        &&  cd release \
        &&  cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. \
        &&  make -j8 \
        &&  make install

# CERES
RUN     wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz        
RUN     tar -xzf ceres-solver-2.2.0.tar.gz
RUN     rm ceres-solver-2.2.0.tar.gz
RUN     cd ceres-solver-2.2.0 \
        &&  mkdir release \
        &&  cd release \
        &&  cmake .. \
        &&  make -j8 \
        &&  make test \
        &&  make install

# Add other programs to end so that it does not build the entire thing
# for launching terminal (for teleop)
RUN apt-get update && apt-get install -y \
	xterm -y \
	ros-jazzy-grid-map -y \
	ros-jazzy-navigation2 -y \
	ros-jazzy-nav2-bringup -y

# Install Python for HMI
RUN apt-get install python3 -y \
		    python3-dev -y \
		    python3-pip -y 

RUN pip3 install tk --break-system-packages
RUN pip3 install tkintermapview --break-system-packages

# Install PyTorch
RUN pip3 install torch torchvision --break-system-packages

# Use root user & define working environment
USER root
RUN mkdir -p /root/ros_jazzy_ws
WORKDIR /root/ros_jazzy_ws

# ROS2 environment setting for root user
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/ros_jazzy_ws/install/setup.bash" >> /root/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash && cd /root/ros_jazzy_ws && colcon build"

RUN echo 'alias start_simu="ros2 launch rover_gz_bringup startSimulator.launch.py"' >> ~/.bashrc

ENV XDG_RUNTIME_DIR=/tmp

# Enter the docker in a terminal
RUN mkdir -p /root/.config/terminator
COPY ../terminatorLayout.txt /root/.config/terminator/config






