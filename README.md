# minimal_examples_crocoddyl
Demo scripts to quickly start with Crocoddyl


# Dependencies
For OCP scripts
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) 
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)

For MPC simulations 
- [PyBullet](https://pybullet.org/wordpress/)
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) 


# Usage
For the reaching task, run `python ocp_kuka_reaching.py` to solve the OCP and visualize / plot the solution. Run `python mpc_kuka_reaching.py` to simulate it in MPC in PyBullet. Same for contact task.

The scripts are as minimal and self-explanatory. The machinery for data extraction and plotting is "hidden" in the utils. 
