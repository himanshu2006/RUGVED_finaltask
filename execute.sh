#!/bin/bash

WORKSPACE=~/RUGVED_finaltask

# Terminal 1
gnome-terminal -- bash -c "
cd $WORKSPACE
colcon build
source install/setup.bash
ros2 launch bot_simulation spdemergency.launch.py
exec bash
"

# Terminal 2
gnome-terminal -- bash -c "
cd $WORKSPACE
colcon build
source install/setup.bash
ros2 run bot_brain shadow_ranger
exec bash
"

# Terminal 3
gnome-terminal -- bash -c "
cd $WORKSPACE
colcon build
source install/setup.bash
ros2 run bot_brain obstacle_avoid_node
exec bash
"

#Run this one time -> chmod +x run_all.sh
