'''
Example script : Crocoddyl OCP with KUKA arm 
static target reaching task
'''

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import ocp_utils
from gnms_merit import GNMS_linesearch

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


# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # endeff frame translation cost
endeff_frame_id = model.getFrameId("contact")
# endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
endeff_translation = np.array([-0.4, 0.3, 0.7]) # move endeff +10 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 10)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 10)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Optionally add armature to take into account actuator's inertia
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Create solver + callbacks

# FDDP
fddp = crocoddyl.SolverFDDP(problem)
fddp.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
xs_init = [x0 for i in range(T+1)]
us_init = fddp.problem.quasiStatic(xs_init[:-1])
fddp.solve([], [], maxiter=20, isFeasible=False)

print('-----')
# GNMS
# ddp = crocoddyl.SolverGNMS(problem)
ddp = GNMS_linesearch(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])
xs_init = [x0 for i in range(T+1)] #fddp.xs #[x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1]) #fddp.us #ddp.problem.quasiStatic(xs_init[:-1])
# ddp.solve(xs_init, us_init, maxiter=20, isFeasible=False)
ddp.solve([], [], maxiter=20, isFeasible=False)

# # Extract DDP data and plot
# ddp_data = ocp_utils.extract_ocp_data(ddp, ee_frame_name='contact')

# ocp_utils.plot_ocp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)

# # Display solution in Gepetto Viewer
# display = crocoddyl.GepettoDisplay(robot)
# display.displayFromSolver(ddp, factor=1)

