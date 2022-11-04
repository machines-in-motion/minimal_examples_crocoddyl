'''
Example script : MPC simulation with KUKA arm 
static target reaching task
'''

import crocoddyl
import numpy as np
np.set_printoptions(precision=4, linewidth=180)

import pin_utils, mpc_utils

from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p


# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Simulation environment
env = BulletEnvWithGround(p.GUI, dt=1e-3)
# Robot simulator 
robot_simulator = IiwaRobot()
# Extract robot model
nq = robot_simulator.pin_robot.model.nq
nv = robot_simulator.pin_robot.model.nv
nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
# Add robot to simulation and initialize
env.add_robot(robot_simulator)
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)
print("[PyBullet] Created robot (id = "+str(robot_simulator.robotId)+")")


# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #
# State and actuation model
state = crocoddyl.StateMultibody(robot_simulator.pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# Create cost terms 
  # Control regularization cost
# uResidual = crocoddyl.ResidualModelControlGrav(state)
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # endeff frame translation cost
endeff_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
# endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
endeff_translation = np.array([0.4, 0.3, 0.7]) # move endeff +10 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
# Add costs
runningCostModel.addCost("stateReg", xRegCost, 1e-1)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-3)
runningCostModel.addCost("translation", frameTranslationCost, 0.5)
terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
terminalCostModel.addCost("translation", frameTranslationCost, 0.5)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 0.04
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
# Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# Create the shooting problem
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                   crocoddyl.CallbackVerbose()])
# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
# Solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)



# # # # # # # # # # # #
###  MPC SIMULATION ###
# # # # # # # # # # # #
# OCP parameters
ocp_params = {}
ocp_params['N_h']          = T
ocp_params['dt']           = dt
ocp_params['maxiter']      = 100 
ocp_params['pin_model']    = robot_simulator.pin_robot.model
ocp_params['armature']     = runningModel.differential.armature
ocp_params['id_endeff']    = endeff_frame_id
ocp_params['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()
# Simu parameters
sim_params = {}
sim_params['sim_freq']  = int(1./env.dt)
sim_params['mpc_freq']  = 0.5 #100 #25   
sim_params['T_sim']     = 2.
log_rate = 100
# Initialize simulation data 
sim_data = mpc_utils.init_sim_data(sim_params, ocp_params, x0)
# Display target 
mpc_utils.display_ball(endeff_translation, RADIUS=.05, COLOR=[1.,0.,0.,.6])
# Simulate
mpc_cycle = 0
ctrl_cycle = 0
# RICCATI_TYPE = 1 
# INTERP_FEEDFORWARD = False
# Kv = 0.*np.eye(nv)
NAIVE = False
for i in range(sim_data['N_sim']): 

    if(i%log_rate==0): 
        print("\n SIMU step "+str(i)+"/"+str(sim_data['N_sim'])+"\n")

    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
    if(i%int(sim_params['sim_freq']/sim_params['mpc_freq']) == 0):
        # Set x0 to measured state 
        ddp.problem.x0 = sim_data['state_mea_SIM_RATE'][i, :]
        # Warm start using previous solution
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['state_mea_SIM_RATE'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 

        # Solve OCP & record MPC predictions
        solved = ddp.solve(xs_init, us_init, maxiter=ocp_params['maxiter'], isFeasible=False)
        print(solved)
        sim_data['state_pred'][mpc_cycle, :, :]  = np.array(ddp.xs)
        sim_data['ctrl_pred'][mpc_cycle, :, :]   = np.array(ddp.us)
        # Extract relevant predictions for interpolations
        x_curr = sim_data['state_pred'][mpc_cycle, 0, :]    # x0* = measured state    (q^,  v^ )
        x_pred = sim_data['state_pred'][mpc_cycle, 1, :]    # x1* = predicted state   (q1*, v1*) 
        u_curr = sim_data['ctrl_pred'][mpc_cycle, 0, :]     # u0* = optimal control   (tau0*)
        u_pred = sim_data['ctrl_pred'][mpc_cycle, 1, :]     # u1* = predicted optimal control (tau1*)
        # Record costs references
        q = sim_data['state_pred'][mpc_cycle, 0, :sim_data['nq']]
        sim_data['ctrl_ref'][mpc_cycle, :]       = pin_utils.get_u_grav(q, ddp.problem.runningModels[0].differential.pinocchio, ocp_params['armature'])
        sim_data['state_ref'][mpc_cycle, :]      = ddp.problem.runningModels[0].differential.costs.costs['stateReg'].cost.residual.reference
        sim_data['lin_pos_ee_ref'][mpc_cycle, :] = ddp.problem.runningModels[0].differential.costs.costs['translation'].cost.residual.reference


        # Select reference control and state for the current MPC cycle
        x_ref_MPC_RATE  = x_curr + sim_data['ocp_to_mpc_ratio'] * (x_pred - x_curr)
        u_ref_MPC_RATE  = u_curr 
        if(mpc_cycle==0):
            sim_data['state_des_MPC_RATE'][mpc_cycle, :]   = x_curr  
        sim_data['ctrl_des_MPC_RATE'][mpc_cycle, :]        = u_ref_MPC_RATE   
        sim_data['state_des_MPC_RATE'][mpc_cycle+1, :]     = x_ref_MPC_RATE    

        xs = ddp.xs.tolist()  
        us = ddp.us.tolist()
        K = ddp.K.tolist()
        for iii in range(T-1):    
          if iii == 0:    
            xs_int = np.linspace(xs[iii].copy(), xs[iii+1].copy(), int(dt/sim_data['dt_sim']))   
            us_int = np.linspace(us[iii].copy(), us[iii+1].copy(), int(dt/sim_data['dt_sim']))    
            Klist = [K[iii].copy()] * int(dt/sim_data['dt_sim'])    
          else:    
            xs_int = np.vstack((xs_int, np.linspace(xs[iii].copy(), xs[iii+1].copy(), int(dt/sim_data['dt_sim']))))    
            us_int = np.vstack((us_int, np.linspace(us[iii].copy(), us[iii+1].copy(), int(dt/sim_data['dt_sim']))))                  
            Klist = Klist + [K[iii].copy()] * int(dt/sim_data['dt_sim'])

        # Increment planning counter
        mpc_cycle += 1
        ctrl_cycle = 0

    # Select reference control and state for the current SIMU cycle
    x_ref_SIM_RATE  = x_curr + ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (x_pred - x_curr)

    if(NAIVE):
      u_ref_SIM_RATE  = u_curr #+ ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (u_pred - u_curr)
    else:  
      u_ref_SIM_RATE  = us_int[ctrl_cycle] #u_curr #+ ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (u_pred - u_curr)
    # if(INTERP_FEEDFORWARD):
    #     u_ref_SIM_RATE += ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (u_pred - u_curr)

    # First prediction = measurement = initialization of MPC
    if(i==0):
        sim_data['state_des_SIM_RATE'][i, :]   = x_curr  
    sim_data['ctrl_des_SIM_RATE'][i, :]        = u_ref_SIM_RATE  
    sim_data['state_des_SIM_RATE'][i+1, :]     = x_ref_SIM_RATE 

    # Interpolate state 
    # if(RICCATI_TYPE == 1):
    # u_ref_SIM_RATE += ddp.K[0] @ (x_ref_SIM_RATE - sim_data['state_mea_SIM_RATE'][i, :])
    # # Use x0
    # elif(RICCATI_TYPE == 2):

    if(NAIVE):
      u_ref_SIM_RATE += ddp.K[0] @ (ddp.problem.x0 - sim_data['state_mea_SIM_RATE'][i, :])  
    else:
      u_ref_SIM_RATE += Klist[ctrl_cycle] @ (xs_int[ctrl_cycle] - sim_data['state_mea_SIM_RATE'][i, :])  

    # # Interpolate gains + state 
    # elif(RICCATI_TYPE == 3):
    # Kinterp = ddp.K[0] + ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (ddp.K[1] - ddp.K[0])
    # u_ref_SIM_RATE += Kinterp @ (x_ref_SIM_RATE - sim_data['state_mea_SIM_RATE'][i, :]) 
    # # Interpolate the whole feedabck 
    # elif(RICCATI_TYPE == 4):
    # ufb0 = ddp.K[0] @ (ddp.problem.x0 - sim_data['state_mea_SIM_RATE'][i, :])
    # ufb1 = ddp.K[1] @ (x_pred - sim_data['state_mea_SIM_RATE'][i, :])
    # ufb = ufb0 + ctrl_cycle*sim_data['ocp_to_sim_ratio'] * (ufb1 - ufb0)
    # u_ref_SIM_RATE += ufb #Kinterp @ (x_ref_SIM_RATE - sim_data['state_mea_SIM_RATE'][i, :]) 
    # else:
    #     u_ref_SIM_RATE -= Kv @ sim_data['state_mea_SIM_RATE'][i, nq:]

    # Send torque to simulator & step simulator
    robot_simulator.send_joint_command(u_ref_SIM_RATE)
    env.step()
    # Measure new state from simulator 
    q_mea_SIM_RATE, v_mea_SIM_RATE = robot_simulator.get_state()
    # Update pinocchio model
    robot_simulator.forward_robot(q_mea_SIM_RATE, v_mea_SIM_RATE)
    # Record data 
    x_mea_SIM_RATE = np.concatenate([q_mea_SIM_RATE, v_mea_SIM_RATE]).T 
    sim_data['state_mea_SIM_RATE'][i+1, :] = x_mea_SIM_RATE
    ctrl_cycle += 1

plot_data = mpc_utils.extract_plot_data_from_sim_data(sim_data)

mpc_utils.plot_mpc_results(plot_data, which_plots=['u', 'x', 'ee_lin'], PLOT_PREDICTIONS=True) #pred_plot_sampling=int(sim_params['mpc_freq']/10))
