'''
Example script : Crocoddyl OCP with KUKA arm with KUKA arm 
static target reaching task while avoiding an obstacle.
Based from the mpc_kuka_reaching.py code. 

Written by Arthur Haffemayer.
'''

import crocoddyl
import numpy as np
import ocp_utils
import pin_utils
import hppfcl
import pinocchio as pin
import mim_solvers
try:
    from colmpc import ResidualDistanceCollision
except:
    print(
        """It needs to be built with COLMPC https://github.com/agimus-project/colmpc
and HPPFCL the devel branch for it to work."""
    )
# # # # # # # # # # # # #
### LOAD ROBOT MODEL  ###
# # # # # # # # # # # # #

from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper("iiwa_convex")

model = robot.model
cmodel = pin_utils.transform_model_into_capsules(robot.collision_model)
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)

# # # # # # # # # # # #
### ADDING OBSTACLE ###
# # # # # # # # # # # #

OBSTACLE_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([-0.2, 0.15, 0.7]))
OBSTACLE = hppfcl.Sphere(1e-1)
OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    0,
    0,
    OBSTACLE,
    OBSTACLE_POSE,
)
ID_OBSTACLE = cmodel.addGeometryObject(
    OBSTACLE_GEOM_OBJECT
)

shapes_avoiding_collision_with_obstacles = [
    "L3_capsule_0",
    "L4_capsule_0",
    "L4_capsule_1",
    "L5_capsule_0",
    "L6_capsule_0",
    "L6_capsule_1",
    "L7_capsule_0",
]

# Adding the collisions pairs to the collision model
for shape in shapes_avoiding_collision_with_obstacles:  
    cmodel.addCollisionPair(
        pin.CollisionPair(
            cmodel.getGeometryId(shape),
            cmodel.getGeometryId("obstacle"),
        )
    )

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Constraint model managers
runningConstraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
terminalConstraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)

# Create collision avoidance constraints

# Creating the residual
if len(cmodel.collisionPairs) != 0:
    for col_idx in range(len(cmodel.collisionPairs)):
        # obstacleDistanceResidual = ResidualCollision(
        #     state, cmodel, cdata, col_idx
        # )
        obstacleDistanceResidual = ResidualDistanceCollision(state, 7, cmodel, col_idx)

        # Creating the inequality constraint
        constraint = crocoddyl.ConstraintModelResidual(
            state,
            obstacleDistanceResidual,
            np.array([2e-2]),
            np.array([np.inf]),
        )

        # Adding the constraint to the constraint manager
        runningConstraintModelManager.addConstraint("col_" + str(col_idx), constraint)
        terminalConstraintModelManager.addConstraint("col_term_" + str(col_idx), constraint)

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


# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 10)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 10)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel, runningConstraintModelManager)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel, terminalConstraintModelManager)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 250
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Create solver + callbacks
solver = mim_solvers.SolverCSQP(problem)
xs_init = [x0 for i in range(T + 1)]
us_init = solver.problem.quasiStatic(xs_init[:-1])
# Solve
solver.termination_tolerance = 1e-4
solver.with_callbacks = True
solver.solve(xs_init, us_init, 100)
solver.with_callbacks = False

# Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(solver, ee_frame_name='contact')
ocp_utils.plot_ocp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)

# Display in Meshcat
import time
from pinocchio.visualize import MeshcatVisualizer

robot.visual_model.addGeometryObject(OBSTACLE_GEOM_OBJECT)
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

viz.initViewer(open=True)
viz.loadViewerModel()
viz.display(q0)
viz.displayCollisions(False)

input("Press Enter to start the simulation")
for t in range(T):
    viz.display(solver.xs[t][:model.nq])
    time.sleep(0.1)
