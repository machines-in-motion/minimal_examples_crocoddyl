
import numpy as np
import pinocchio as pin
import eigenpy
from numpy.linalg import pinv
import time
import matplotlib.pyplot as plt


# Rotate placement
def rotate(se3_placement, rpy=[0., 0., 0.]):
    '''
    Rotates se3_placement.rotation by rpy (LOCAL)
     input : 
        se3_placement : pin.SE3
        rpy           : RPY orientation in LOCAL frame
                        RPY       
    '''
    se3_placement_rotated = se3_placement.copy()
    R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    se3_placement_rotated.rotation = se3_placement_rotated.rotation.copy().dot(R)
    return se3_placement_rotated

    

# Get frame position
def get_p(q, pin_robot, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_p_(q, pin_robot.model, id_endeff)

def get_p_(q, model, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    
    data = model.createData()
    if(type(q)==np.ndarray and len(q.shape)==1):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[id_endeff].translation.T
    else:
        N = np.shape(q)[0]
        p = np.empty((N,3))
        for i in range(N):
            pin.forwardKinematics(model, data, q[i])
            pin.updateFramePlacements(model, data)
            p[i,:] = data.oMf[id_endeff].translation.T
    return p



# Get frame linear velocity
def get_v(q, dq, pin_robot, id_endeff, ref=pin.LOCAL):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_v_(q, dq, pin_robot.model, id_endeff, ref)

def get_v_(q, dq, model, id_endeff, ref=pin.LOCAL):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    data = model.createData()
    if(len(q) != len(dq)):
        logger.error("q and dq must have the same size !")
    if(type(q)==np.ndarray and len(q.shape)==1):
        # J = pin.computeFrameJacobian(model, data, q, id_endeff)
        # v = J.dot(dq)[:3] 
        pin.forwardKinematics(model, data, q, dq)
        spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
        v = spatial_vel.linear
    else:
        N = np.shape(q)[0]
        v = np.empty((N,3))
        for i in range(N):
            # J = pin.computeFrameJacobian(model, data, q[i,:], id_endeff)
            # v[i,:] = J.dot(dq[i])[:3] 
            pin.forwardKinematics(model, data, q[i], dq[i])
            spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
            v[i,:] = spatial_vel.linear    
    return v



# Get frame orientation (rotation)
def get_R(q, pin_robot, id_endeff):
    '''
    Returns end-effector rotation matrices given q trajectory 
        q         : joint positions
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_R_(q, pin_robot.model, id_endeff)

def get_R_(q, model, id_endeff):
    '''
    Returns end-effector rotation matrices given q trajectory
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    Output : single 3x3 array (or list of 3x3 arrays)
    '''
    data = model.createData()
    if(type(q)==np.ndarray and len(q.shape)==1):
        pin.framesForwardKinematics(model, data, q)
        R = data.oMf[id_endeff].rotation.copy()
    else:
        N = np.shape(q)[0]
        R = []    
        for i in range(N):    
            pin.framesForwardKinematics(model, data, q[i])
            R.append(data.oMf[id_endeff].rotation.copy())
    return R



# Get frame orientation (RPY)
def get_rpy(q, pin_robot, id_endeff):
    '''
    Returns RPY angles of end-effector frame given q trajectory
        q         : joint positions
        model     : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_rpy_(q, pin_robot.model, id_endeff)

def get_rpy_(q, model, id_endeff):
    '''
    Returns RPY angles of end-effector frame given q trajectory
        q         : joint positions
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    R = get_R_(q, model, id_endeff)
    if(type(R)==list):
        N = np.shape(q)[0]
        rpy = np.empty((N,3))
        for i in range(N):
            rpy[i,:] = pin.rpy.matrixToRpy(R[i]) #%(2*np.pi)
    else:
        rpy = pin.rpy.matrixToRpy(R) #%(2*np.pi)
    return rpy



# Get frame angular velocity
def get_w(q, dq, pin_robot, id_endeff, ref=pin.LOCAL):
    '''
    Returns end-effector angular velocity given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    return get_w_(q, dq, pin_robot.model, id_endeff, ref)

def get_w_(q, dq, model, id_endeff, ref=pin.LOCAL):
    '''
    Returns end-effector  angular velocity given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        pin_robot : pinocchio wrapper
        id_endeff : id of EE frame
    '''
    data = model.createData()
    if(len(q) != len(dq)):
        logger.error("q and dq must have the same size !")
    if(type(q)==np.ndarray and len(q.shape)==1):
        pin.forwardKinematics(model, data, q, dq)
        spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
        w = spatial_vel.angular
    else:
        N = np.shape(q)[0]
        w = np.empty((N,3))
        for i in range(N):
            pin.forwardKinematics(model, data, q[i], dq[i])
            spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
            w[i,:] = spatial_vel.angular    
    return w



# Get frame force
def get_f_(q, v, tau, model, id_endeff, armature, REG=0.):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = -pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        pin.computeJointJacobians(model, data)
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Joint space inertia and its inverse + NL terms
        pin.computeAllTerms(model, data, q[i,:], v[i,:])
        data.M += np.diag(armature)
        Minv = np.linalg.inv(data.M)
        h = pin.nonLinearEffects(model, data, q[i,:], v[i,:])
        # Contact force
        # f = (JMiJ')^+ ( JMi (b-tau) + gamma )
        REGMAT = REG*np.eye(6)
        LDLT = eigenpy.LDLT(J @ Minv @ J.T + REGMAT)
        f[i,:]  = LDLT.solve(J @ Minv @ (h - tau[i,:]) + gamma.vector)
        # f[i,:] = np.linalg.solve( J @ Minv @ J.T + REGMAT,  J @ Minv @ (h - tau[i,:]) + gamma.vector )
    return f

def get_f_lambda(q, v, tau, model, id_endeff, armature, REG=0.):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.computeJointJacobians(model, data, q[i,:])
        pin.framesForwardKinematics(model, data, q[i,:])
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
          # Forward kinematics & placements
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        # Joint space inertia and its inverse + NL terms
        # pin.computeAllTerms(model, data, q[i,:], v[i,:])
        data.M += np.diag(armature)
        pin.forwardDynamics(model, data, q[i,:], v[i,:], tau[i,:], J[:6,:], gamma.vector, REG)
        # Contact force
        f[i,:] = data.lambda_c
    return f

def get_f_kkt(q, v, tau, model, id_endeff):
    '''
    Returns contact force in LOCAL frame based on FD estimate of joint acc
        q         : joint positions
        v         : joint velocities
        a         : joint acceleration
        tau       : joint torques
        pin_robot : Pinocchio wrapper
        id_endeff : id of EE frame
        dt        : step size for FD estimate of joint acceleration
    '''
    data = model.createData()
    # Calculate contact force from (q, v, a, tau)
    f = np.empty((q.shape[0]-1, 6))
    for i in range(f.shape[0]):
        # Get spatial acceleration at EE frame
        pin.computeJointJacobians(model, data, q[i,:])
        pin.framesForwardKinematics(model, data, q[i,:])
        J = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
          # Forward kinematics & placements
        pin.forwardKinematics(model, data, q[i,:], v[i,:], np.zeros(q.shape[1]))
        pin.updateFramePlacements(model, data)
        gamma = pin.getFrameAcceleration(model, data, id_endeff, pin.ReferenceFrame.LOCAL)
        # Joint space inertia and its inverse + NL terms
        h = pin.nonLinearEffects(model, data, q[i,:], v[i,:])
        rhs = np.vstack([np.array([h - tau[i,:]]).T, np.array([gamma.vector]).T ])
        f[i,:] = pin.computeKKTContactDynamicMatrixInverse(model, data, q[i,:], J).dot(rhs)[-6:,0]
    return f



# Get gravity joint torque
def get_u_grav(q, model, armature):
    '''
    Return gravity torque at q
    '''
    data = model.createData()
    data.M += np.diag(armature)
    return pin.computeGeneralizedGravity(model, data, q)

# Get joint torques 
def get_tau(q, v, a, f, model, armature):
    '''
    Return torque using rnea
    '''
    data = model.createData()
    data.M += np.diag(armature)
    return pin.rnea(model, data, q, v, a, f)

# Get joint torques due to an external wrench 
def get_external_joint_torques(M_contact, wrench, robot):
    '''
    Computes the torques induced at each joint by an external contact wrench
    '''
    f_ext = []
    if(type(wrench)=='list'):
        wrench = np.array(wrench)
    # Compute joint torques due to desired external force 
    for i in range(robot.model.nq+1):
        # CONTACT --> WORLD
        W_M_ct = M_contact.copy()
        f_WORLD = W_M_ct.actionInverse.T.dot(wrench)
        # WORLD --> JOINT
        j_M_W = robot.data.oMi[i].copy().inverse()
        f_JOINT = j_M_W.actionInverse.T.dot(f_WORLD)
        f_ext.append(pin.Force(f_JOINT))
    return f_ext




# Inverse kinematics
def IK_position(robot, q, frame_id, p_des, LOGS=False, DISPLAY=False, DT=1e-2, IT_MAX=1000, EPS=1e-6, sleep=0.01):
    '''
    Inverse kinematics: returns q, v to reach desired position p
    '''
    errs =[]
    for i in range(IT_MAX):  
        if(i%10 == 0 and LOGS==True):
            print("Step "+str(i)+"/"+str(IT_MAX))
        pin.framesForwardKinematics(robot.model, robot.data, q)  
        oMtool = robot.data.oMf[frame_id]          
        oRtool = oMtool.rotation                  
        tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, frame_id)
        o_Jtool3 = oRtool.dot( tool_Jtool[:3,:] )         # 3D Jac of EE in W frame
        o_TG = oMtool.translation - p_des                 # translation err in W frame 
        vq = -pinv(o_Jtool3).dot(o_TG)                    # vel in negative err dir
        q = pin.integrate(robot.model,q, vq * DT)         # take step
        if(DISPLAY):
            robot.display(q)                                   
            time.sleep(sleep)
        errs.append(o_TG)
        if(i%10 == 0 and LOGS==True):
            print(np.linalg.norm(o_TG))
        if np.linalg.norm(o_TG) < EPS:
            break    
    return q, vq, errs

def IK_placement(robot, q0, frame_id, oMf_des, LOGS=False, DT=1e-2, IT_MAX=1000, EPS=1e-6, DAMP=1e-6):
    '''
    Inverse kinematics: returns q, v to reach desired placement M 
    '''
    data = robot.data 
    model = robot.model
    q = q0.copy()
    vq = np.zeros(model.nq)
    pin.framesForwardKinematics(model, data, q)
    oMf = data.oMf[frame_id]
    errs = []
    # Loop on an inverse kinematics for 200 iterations.
    for i in range(IT_MAX): 
        if(i%10 == 0 and LOGS==True):
            print("Step "+str(i)+"/"+str(IT_MAX))
        pin.framesForwardKinematics(model, data, q)  
        dMi = oMf_des.actInv(oMf)
        err = pin.log(dMi).vector
        errs.append(err)
        if(i%10 == 0 and LOGS==True):
            print(np.linalg.norm(err))
        J = pin.computeFrameJacobian(model, data, q, frame_id)    
        vq = - J.T @ pinv(J.dot(J.T) + DAMP * np.eye(6)) @ err    
        # vq = - J.T.dot(np.linalg.solve(J.dot(J.T) + DAMP * np.eye(6), err))
        q = pin.integrate(model, q, vq * DT)
    return q, vq, errs


def extract_ddp_data(ddp, ee_frame_name='contact', ct_frame_name='contact'): 
    '''
    Record relevant data from ddp solver in order to plot 
    ee_frame_name = name of frame for which ee plots will be generated
                        by default 'contact' as in KUKA urdf model (Tennis ball)
    ct_frame_name = name of frame for which force plots will be generated
                        by default 'contact' as in KUKA urdf model (Tennis ball)
    '''
    # Store data
    ddp_data = {}
    # OCP params
    ddp_data['T'] = ddp.problem.T
    ddp_data['dt'] = ddp.problem.runningModels[0].dt
    ddp_data['nq'] = ddp.problem.runningModels[0].state.nq
    ddp_data['nv'] = ddp.problem.runningModels[0].state.nv
    ddp_data['nu'] = ddp.problem.runningModels[0].differential.actuation.nu
    ddp_data['nx'] = ddp.problem.runningModels[0].state.nx
    # Pin model
    ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
    ddp_data['armature'] = ddp.problem.runningModels[0].differential.armature
    ddp_data['frame_id'] = ddp_data['pin_model'].getFrameId(ee_frame_name)
    # Solution trajectories
    ddp_data['xs'] = ddp.xs
    ddp_data['us'] = ddp.us
    ddp_data['CONTACT_TYPE'] = None
    # Extract force at EE frame and contact info 
    if(hasattr(ddp.problem.runningModels[0].differential, 'contacts')):
      # Get refs for contact model
      contactModelRef0 = ddp.problem.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.reference
      # Case 6D contact (x,y,z,Ox,Oy,Oz)
      if(hasattr(contactModelRef0, 'rotation')):
        ddp_data['contact_rotation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.rotation for i in range(ddp.problem.T)]
        ddp_data['contact_rotation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.rotation)
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.translation for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.translation)
        ddp_data['CONTACT_TYPE'] = '6D'
      # Case 3D contact (x,y,z)
      elif(np.size(contactModelRef0)==3):
        # Get ref translation for 3D 
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
        ddp_data['CONTACT_TYPE'] = '3D'
      # Case 1D contact (z)
      elif(np.size(contactModelRef0)==1):
        ddp_data['contact_translation'] = [ddp.problem.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(ddp.problem.T)]
        ddp_data['contact_translation'].append(ddp.problem.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
        ddp_data['CONTACT_TYPE'] = '1D'
      # Get contact force
      datas = [ddp.problem.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(ddp.problem.T)]
      # data.f = force exerted at parent joint expressed in WORLD frame (oMi)
      # express it in LOCAL contact frame using jMf 
      ee_forces = [data.jMf.actInv(data.f).vector for data in datas] 
      ddp_data['fs'] = [ee_forces[i] for i in range(ddp.problem.T)]
    # Extract refs for active costs 
    # TODO : active costs may change along horizon : how to deal with that when plotting? 
    ddp_data['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()
    if('stateReg' in ddp_data['active_costs']):
        ddp_data['stateReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['stateReg_ref'].append(ddp.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    if('ctrlReg' in ddp_data['active_costs']):
        ddp_data['ctrlReg_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlReg'].cost.residual.reference for i in range(ddp.problem.T)]
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ddp_data['ctrlRegGrav_ref'] = [get_u_grav(ddp.xs[i][:ddp_data['nq']], ddp_data['pin_model'], ddp_data['armature']) for i in range(ddp.problem.T)]
    if('stateLim' in ddp_data['active_costs']):
        ddp_data['stateLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
        ddp_data['stateLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
        ddp_data['stateLim_ub'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.ub)
        ddp_data['stateLim_lb'].append(ddp.problem.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.lb)
    if('ctrlLim' in ddp_data['active_costs']):
        ddp_data['ctrlLim_ub'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub for i in range(ddp.problem.T)]
        ddp_data['ctrlLim_lb'] = [ddp.problem.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb for i in range(ddp.problem.T)]
        ddp_data['ctrlLim_ub'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub)
        ddp_data['ctrlLim_lb'].append(ddp.problem.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb)
    if('placement' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.translation for i in range(ddp.problem.T)]
        ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation)
        ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.rotation for i in range(ddp.problem.T)]
        ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.rotation)
    if('translation' in ddp_data['active_costs']):
        ddp_data['translation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['translation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    if('velocity' in ddp_data['active_costs']):
        ddp_data['velocity_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(ddp.problem.T)]
        ddp_data['velocity_ref'].append(ddp.problem.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
        # ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['velocity'].cost.residual.id
    if('rotation' in ddp_data['active_costs']):
        ddp_data['rotation_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['rotation'].cost.residual.reference for i in range(ddp.problem.T)]
        ddp_data['rotation_ref'].append(ddp.problem.terminalModel.differential.costs.costs['rotation'].cost.residual.reference)
    if('force' in ddp_data['active_costs']): 
        ddp_data['force_ref'] = [ddp.problem.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(ddp.problem.T)]
    return ddp_data


# Plot from DDP solver 
def plot_ddp_results(DDP_DATA, which_plots='all', labels=None, markers=None, colors=None, sampling_plot=1, SHOW=False):
    '''
    Plot ddp results from 1 or several DDP solvers
        X, U, EE trajs
        INPUT 
        DDP_DATA         : DDP data or list of ddp data (cf. data_utils.extract_ddp_data())
    '''
    if(type(DDP_DATA) != list):
        DDP_DATA = [DDP_DATA]
    if(labels==None):
        labels=[None for k in range(len(DDP_DATA))]
    if(markers==None):
        markers=[None for k in range(len(DDP_DATA))]
    if(colors==None):
        colors=[None for k in range(len(DDP_DATA))]
    for k,data in enumerate(DDP_DATA):
        # If last plot, make legend
        make_legend = False
        if(k+sampling_plot > len(DDP_DATA)-1):
            make_legend=True
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_x, ax_x = plot_ddp_state(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('us' in data.keys()):
                    fig_u, ax_u = plot_ddp_control(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('xs' in data.keys()):
                    fig_ee_lin, ax_ee_lin = plot_ddp_endeff_linear(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                    fig_ee_ang, ax_ee_ang = plot_ddp_endeff_angular(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
            if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                if('fs' in data.keys()):
                    fig_f, ax_f = plot_ddp_force(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_state(data, fig=fig_x, ax=ax_x, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('us' in data.keys()):
                        plot_ddp_control(data, fig=fig_u, ax=ax_u, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('xs' in data.keys()):
                        plot_ddp_endeff_linear(data, fig=fig_ee_lin, ax=ax_ee_lin, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                    if('fs' in data.keys()):
                        plot_ddp_force(data, fig=fig_f, ax=ax_f, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
    if(SHOW):
      plt.show()
    
    # Record and return if user needs to overlay stuff
    fig = {}
    ax = {}
    if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['x'] = fig_x
            ax['x'] = ax_x
    if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('us' in data.keys()):
            fig['u'] = fig_u
            ax['u'] = ax_u
    if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('xs' in data.keys()):
            fig['ee_lin'] = fig_ee_lin
            ax['ee_lin'] = ax_ee_lin
            fig['ee_ang'] = fig_ee_ang
            ax['ee_ang'] = ax_ee_ang
    if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
        if('fs' in data.keys()):
            fig['f'] = fig_f
            ax['f'] = ax_f

    return fig, ax

def plot_ddp_state(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq'] 
    nv = ddp_data['nv'] 
    # Extract trajectories
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nv:]
    # If state reg cost, 
    if('stateReg' in ddp_data['active_costs']):
        x_reg_ref = np.array(ddp_data['stateReg_ref'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col') 
    if(label is None):
        label='State'
    for i in range(nq):
        # Plot positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot joint position regularization reference
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,0].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,0].plot(tspan, x_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)  

        # Plot joint velocity regularization reference
        if('stateReg' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reg_ref' in labels):
                handles.pop(labels.index('reg_ref'))
                ax[i,1].lines.pop(labels.index('reg_ref'))
                labels.remove('reg_ref')
            ax[i,1].plot(tspan, x_reg_ref[:,nq+i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        
        # Labels, tick labels and grid
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)  

    # Common x-labels + align
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nu = ddp_data['nu'] 
    # Extract trajectory
    u = np.array(ddp_data['us'])
    if('ctrlReg' in ddp_data['active_costs']):
        ureg_ref  = np.array(ddp_data['ctrlReg_ref']) 
    if('ctrlRegGrav' in ddp_data['active_costs']):
        ureg_grav = np.array(ddp_data['ctrlRegGrav_ref'])

    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col') 
    if(label is None):
        label='Control'    

    for i in range(nu):
        # Plot optimal control trajectory
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot control regularization reference 
        if('ctrlReg' in ddp_data['active_costs']):
            handles, labels = ax[i].get_legend_handles_labels()
            if('u_reg' in labels):
                handles.pop(labels.index('u_reg'))
                ax[i].lines.pop(labels.index('u_reg'))
                labels.remove('u_reg')
            ax[i].plot(tspan, ureg_ref[:,i], linestyle='-.', color='k', marker=None, label='u_reg', alpha=0.5)

        # Plot gravity compensation torque
        if('ctrlRegGrav' in ddp_data['active_costs']):
            handles, labels = ax[i].get_legend_handles_labels()
            if('grav(q)' in labels):
                handles.pop(labels.index('u_grav(q)'))
                ax[i].lines.pop(labels.index('u_grav(q)'))
                labels.remove('u_grav(q)')
            ax[i].plot(tspan, ureg_grav[:,i], linestyle='-.', color='m', marker=None, label='u_grav(q)', alpha=0.5)
        
        # Labels, tick labels, grid
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i].grid(True)

    # Set x label + align
    ax[-1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:])
    # Legend
    if(MAKE_LEGEND):
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Control trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff_linear(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                    MAKE_LEGEND=False, SHOW=True, AUTOSCALE=True):
    '''
    Plot ddp results (endeff linear position, velocity)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq']
    nv = ddp_data['nv'] 
    # Extract EE traj
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nq:nq+nv]
    lin_pos_ee = get_p_(q, ddp_data['pin_model'], ddp_data['frame_id'])
    lin_vel_ee = get_v_(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Cost reference frame translation if any, or initial one
    if('translation' in ddp_data['active_costs'] or 'placement' in ddp_data['active_costs']):
        lin_pos_ee_ref = np.array(ddp_data['translation_ref'])
    else:
        lin_pos_ee_ref = np.array([lin_pos_ee[0,:] for i in range(N+1)])
    # Cost reference frame linear velocity if any, or initial one
    if('velocity' in ddp_data['active_costs']):
        lin_vel_ee_ref = np.array(ddp_data['velocity_ref'])[:,:3] # linear part
    else:
        lin_vel_ee_ref = np.array([lin_vel_ee[0,:] for i in range(N+1)])
    # Contact reference translation if CONTACT
    if(ddp_data['CONTACT_TYPE'] is not None):
        lin_pos_ee_contact = np.array(ddp_data['contact_translation'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='OCP solution'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i,0].plot(tspan, lin_pos_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # Plot EE target frame translation in WORLD frame
        if('translation' or 'placement' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        # Plot CONTACT reference frame translation in WORLD frame
        if(ddp_data['CONTACT_TYPE'] is not None):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('Baumgarte stab. ref.' in labels):
                handles.pop(labels.index('Baumgarte stab. ref.'))
                ax[i,0].lines.pop(labels.index('Baumgarte stab. ref.'))
                labels.remove('Baumgarte stab. ref.')
            # Exception for 1D contact: plot only along z-axis 
            if(ddp_data['CONTACT_TYPE']=='1D'):
                if(i==2):
                    ax[i,0].plot(tspan, lin_pos_ee_contact, linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)
            else:
                ax[i,0].plot(tspan, lin_pos_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)
        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot EE (linear) velocities in WORLD frame
        ax[i,1].plot(tspan, lin_vel_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
        # Plot EE target frame (linear) velocity in WORLD frame
        if('velocity' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, lin_vel_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        # Labels, tick labels, grid
        ax[i,1].set_ylabel('$V^{EE}_%s$ (m/s)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
    
    #x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)

    # Set ylim if any
    if(AUTOSCALE):
        TOL = 0.1
        ax_p_ylim = 1.1*max(np.max(np.abs(lin_pos_ee)), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(lin_vel_ee)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(lin_pos_ee_ref[0,i]-ax_p_ylim, lin_pos_ee_ref[0,i]+ax_p_ylim) 
            ax[i,1].set_ylim(lin_vel_ee_ref[0,i]-ax_v_ylim, lin_vel_ee_ref[0,i]+ax_v_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[2,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector frame position and linear velocity', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff_angular(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                    MAKE_LEGEND=False, SHOW=True, AUTOSCALE=False):
    '''
    Plot ddp results (endeff angular position, velocity)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    nq = ddp_data['nq']
    nv = ddp_data['nv'] 
    # Extract EE traj
    x = np.array(ddp_data['xs'])
    q = x[:,:nq]
    v = x[:,nq:nq+nv]
    rpy_ee = get_rpy_(q, ddp_data['pin_model'], ddp_data['frame_id'])
    w_ee   = get_w_(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Cost reference frame orientation if any, or initial one
    if('rotation' in ddp_data['active_costs'] or 'placement' in ddp_data['active_costs']):
        rpy_ee_ref = np.array([pin.utils.matrixToRpy(np.array(R)) for R in ddp_data['rotation_ref']])
    else:
        rpy_ee_ref = np.array([rpy_ee[0,:] for i in range(N+1)])
    # Cost reference angular velocity if any, or initial one
    if('velocity' in ddp_data['active_costs']):
        w_ee_ref = np.array(ddp_data['velocity_ref'])[:,3:] # angular part
    else:
        w_ee_ref = np.array([w_ee[0,:] for i in range(N+1)])
    # Contact reference orientation (6D)
    if(ddp_data['CONTACT_TYPE']=='6D'):
        rpy_ee_contact = np.array([pin.utils.matrixToRpy(R) for R in ddp_data['contact_rotation']])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='OCP solution'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE orientation in WORLD frame
        ax[i,0].plot(tspan, rpy_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame orientation in WORLD frame
        if('rotation' or 'placement' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, rpy_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Plot CONTACT reference frame rotation in WORLD frame
        if(ddp_data['CONTACT_TYPE']=='6D'):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('contact' in labels):
                handles.pop(labels.index('contact'))
                ax[i,0].lines.pop(labels.index('contact'))
                labels.remove('contact')
            ax[i,0].plot(tspan, rpy_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)

        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$RPY^{EE}_%s$ (rad)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot EE 'linear) velocities in WORLD frame
        ax[i,1].plot(tspan, w_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot EE target frame (linear) velocity in WORLD frame
        if('velocity' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, w_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
        
        # Labels, tick labels, grid
        ax[i,1].set_ylabel('$W^{EE}_%s$ (rad/s)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
    
    #x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)

    # Set ylim if any
    if(AUTOSCALE):
        TOL = 0.1
        ax_p_ylim = 1.1*max(np.max(np.abs(rpy_ee)), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(w_ee)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, +ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, +ax_v_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector frame orientation and angular velocity', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_force(ddp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                MAKE_LEGEND=False, SHOW=True, AUTOSCALE=True):
    '''
    Plot ddp results (force)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt']
    # Extract EE traj
    f = np.array(ddp_data['fs'])
    f_ee_lin = f[:,:3]
    f_ee_ang = f[:,3:]
    # Get desired contact wrench (linear, angular)
    if('force' in ddp_data['active_costs']):
        f_ee_ref = np.array(ddp_data['force_ref'])
    else:
        f_ee_ref = np.zeros((N,6))
    f_ee_lin_ref = f_ee_ref[:,:3]
    f_ee_ang_ref = f_ee_ref[:,3:]
    # Plots
    tspan = np.linspace(0, N*dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col')
    if(label is None):
        label='End-effector force'
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot contact linear wrench (force) in LOCAL frame
        ax[i,0].plot(tspan, f_ee_lin[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot desired contact linear wrench (force) in LOCAL frame 
        if('force' in ddp_data['active_costs']):
            handles, labels = ax[i,0].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,0].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,0].plot(tspan, f_ee_lin_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)
        
        # Labels, tick labels+ grid
        ax[i,0].set_ylabel('$\\lambda^{lin}_%s$ (N)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)

        # Plot contact angular wrench (torque) in LOCAL frame 
        ax[i,1].plot(tspan, f_ee_ang[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

        # Plot desired contact anguler wrench (torque) in LOCAL frame
        if('force' in ddp_data['active_costs']):
            handles, labels = ax[i,1].get_legend_handles_labels()
            if('reference' in labels):
                handles.pop(labels.index('reference'))
                ax[i,1].lines.pop(labels.index('reference'))
                labels.remove('reference')
            ax[i,1].plot(tspan, f_ee_ang_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)

        # Labels, tick labels+ grid
        ax[i,1].set_ylabel('$\\lambda^{ang}_%s$ (Nm)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)
    
    # x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)

    # Set ylim if any
    if(AUTOSCALE):
        TOL = 1e-1
        ax_lin_ylim = 1.1*max(np.max(np.abs(f_ee_lin)), TOL)
        ax_ang_ylim = 1.1*max(np.max(np.abs(f_ee_ang)), TOL)
        for i in range(3):
            ax[i,0].set_ylim(f_ee_lin_ref[0,i]-ax_lin_ylim, f_ee_lin_ref[0,i]+ax_lin_ylim) 
            ax[i,1].set_ylim(f_ee_ang_ref[0,i]-ax_ang_ylim, f_ee_ang_ref[0,i]+ax_ang_ylim)

    if(MAKE_LEGEND):
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector forces: linear and angular', size=18)
    if(SHOW):
        plt.show()
    return fig, ax


