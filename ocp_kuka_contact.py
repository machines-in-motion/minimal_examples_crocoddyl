'''
Example script : Crocoddyl OCP with KUKA arm 
contact force task 
'''

import crocoddyl
import pinocchio
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import utils

# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

# # Load robot model directly from URDF & mesh files
# from pinocchio.robot_wrapper import RobotWrapper
# urdf_path = '/home/skleff/robot_properties_kuka/urdf/iiwa.urdf'
# mesh_path = '/home/skleff/robot_properties_kuka'
# robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_path) 

# Or use robot_properties_kuka 
from robot_properties_kuka.config import IiwaConfig
robot = IiwaConfig.buildRobotWrapper()

model = robot.model
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)


# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Contact model 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)

# Create 3D contact on the en-effector frame
contact_frame_id = model.getFrameId("contact")
contact_position = robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 50.])
contact3d = crocoddyl.ContactModel6D(state, contact_frame_id, contact_position, baumgarte_gains) 

# Populate contact model with contacts
contactModel.addContact("contact", contact3d, active=True)


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # End-effector frame force cost
desired_wrench = np.array([0., 0., -20., 0., 0., 0.])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pinocchio.Force(desired_wrench), 6, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)

# Populate cost models with cost terms
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True)
terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Optionally add armature to take into account actuator's inertia
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 250
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])

# Solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)


# Extract DDP data and plot
ddp_data = {}
ddp_data = utils.extract_ddp_data(ddp, ee_frame_name='contact', ct_frame_name='contact')

utils.plot_ddp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)



# Display solution in Gepetto Viewer
display = crocoddyl.GepettoDisplay(robot, frameNames=['contact'])
display.displayFromSolver(ddp, factor=1)


