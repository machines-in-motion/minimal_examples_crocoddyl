
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
