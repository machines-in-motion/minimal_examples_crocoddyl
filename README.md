# minimal_examples_crocoddyl
Demo scripts to quickly get started with Crocoddyl and MPC


# Dependencies
For OCP scripts
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) 
- [matplotlib](https://matplotlib.org/)

For MPC simulations 
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) 


# Usage
For the reaching task, run `python ocp_kuka_reaching.py` to solve the OCP and visualize / plot the solution. Run `python mpc_kuka_reaching.py` to simulate it in MPC in PyBullet. Same for contact task.

The scripts are minimal and self-explanatory. The machinery for data extraction and plotting is hidden in the utils. 

# Copyrights
Copyright(c) 2019-2023 New York University

# License
BSD 3-Clause License
