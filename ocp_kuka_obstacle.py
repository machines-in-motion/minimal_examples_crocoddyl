'''
Example script : Crocoddyl OCP with KUKA arm 
static target reaching task
'''

import crocoddyl
import numpy as np
import pinocchio as pin
import ocp_utils
try:
    import hppfcl
except:
    print("Need to install hpp-flc")

# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #


from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper("iiwa")

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
endeff_translation = np.array([-0.4, 0.3, 0.7]) # move endeff +10 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
  
# COLLISION COST 
# Create a capsule for the arm
link_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
ig_link_names = []
for i,ln in enumerate(link_names):
    pin_link_id         = model.getFrameId(ln)
    pin_joint_id        = model.getJointId(ln)
    geomModel           = pin.GeometryModel() 
    ig_link_names       = geomModel.addGeometryObject(pin.GeometryObject("arm_link_"+str(i), 
                                                      pin_link_id, 
                                                      model.frames[model.getFrameId(ln)].parent,
                                                      hppfcl.Capsule(0, 0.5),
                                                      pin.SE3.Identity()))

# Create obstacle in the world
obsPose = pin.SE3.Identity()
obsPose.translation = np.array([0., 0., 1.])
obsObj = pin.GeometryObject("obstacle",
                             model.getFrameId("universe"),
                             model.frames[model.getFrameId("universe")].parent,
                             hppfcl.Box(0.1, 0.1, 0.1),
                             obsPose)
ig_obs = geomModel.addGeometryObject(obsObj)
geomModel.addCollisionPair(pin.CollisionPair(ig_link_names,ig_obs))
# Add collision cost to the OCP 
collision_radius = 0.2
activationCollision = crocoddyl.ActivationModel2NormBarrier(3, collision_radius)
residualCollision = crocoddyl.ResidualModelPairCollision(state, nu, geomModel, 0, pin_joint_id)
costCollision = crocoddyl.CostModelResidual(state, activationCollision, residualCollision)


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 10)
runningCostModel.addCost("collision", costCollision, 100)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 10)
terminalCostModel.addCost("collision", costCollision, 100)


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
ddp.solve(xs_init, us_init, maxiter=100, is_feasible=False)

# # Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(ddp, ee_frame_name='contact')
ocp_utils.plot_ocp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)

# Display in Meshcat
import time
from pinocchio.visualize import MeshcatVisualizer
robot.visual_model.addGeometryObject(obsObj)

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.display(q0)

viz.displayCollisions(True)
viz.displayVisuals(True)

input("Press Enter to start the simulation")
for t in range(T):
    viz.display(ddp.xs[t][:model.nq])
    time.sleep(0.1)