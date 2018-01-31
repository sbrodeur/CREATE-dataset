
## Compiling ROS on the Beaglebone Black

Install bootstrap dependencies:
```
sudo pip install -U rosdep rosinstall_generator wstool rosinstall
sudo pip install --upgrade setuptools
```

Initialize rosdep:
```
sudo rm -rf /etc/ros/rosdep/sources.list.d/20-default.list
sudo rosdep init
rosdep update
```

Build the catkin packages and install at default location (/opt/ros/kinetic):
```
mkdir -p $HOME/build
cd $HOME/build
mkdir ros_catkin_ws_kinetic
cd ros_catkin_ws_kinetic
rosinstall_generator ros_comm --rosdistro kinetic --deps --wet-only --tar > kinetic-ros_comm-wet.rosinstall
wstool init src kinetic-ros_comm-wet.rosinstall
rosinstall_generator diagnostics image_common image_transport_plugins common_msgs rosconsole_bridge --rosdistro kinetic --deps --wet-only --tar > add-pkgs.rosinstall
wstool merge -t src add-pkgs.rosinstall
wstool update -t src
rosdep install --from-paths src --ignore-src --rosdistro kinetic -y
CATKIN_COMPILE_FLAGS="\"-O3 -save-temps --param ggc-min-expand=10 --param ggc-min-heapsize=4096\""
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=$CATKIN_COMPILE_FLAGS -DCMAKE_CXX_FLAGS=$CATKIN_COMPILE_FLAGS --install-space /opt/ros/kinetic
```

Note that since we compile on a system with only 512MB of RAM, we need to configure gcc gargabe collector to be aggresive and save temporary files to disk.


Set environment variables:
```
echo "source /opt/ros/kinetic/setup.bash" >>  $HOME/.profile
```

For more information on compiling ROS from source, see <http://wiki.ros.org/kinetic/Installation/Source>

## Compiling the code

Download the source code from the git repository:
```
git clone https://github.com/sbrodeur/ros-icreate-bbb.git
```

Create a symlink to the ROS distribution for the catkin workspace (here for ROS kinetic):
```
cd ros-icreate-bbb
ln -s /opt/ros/kinetic/share/catkin/cmake/toplevel.cmake src/CMakeLists.txt
```

Compile workspace with catkin:
```
CATKIN_COMPILE_FLAGS="\"-O3 -save-temps --param ggc-min-expand=10 --param ggc-min-heapsize=4096\""
alias catkin_make="catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=$CATKIN_COMPILE_FLAGS -DCMAKE_CXX_FLAGS=$CATKIN_COMPILE_FLAGS"
catkin_make
```
