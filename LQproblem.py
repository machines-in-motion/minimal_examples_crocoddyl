import numpy as np
import crocoddyl
import matplotlib.pyplot as plt

LINE_WIDTH = 100


class PointMassDynamics:
    def __init__(self):
        self.g = np.array([0.0, -9.81])
        self.mass = 1
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        self.nu = 2
        self.c_drag = 0.0

    def nonlinear_dynamics(self, x, u):
        return (1 / self.mass) * u + self.g - self.c_drag * x[2:] ** 2

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u"""
        dfdx = np.zeros([self.nv, self.ndx])
        dfdu = np.zeros([self.nv, self.nu])
        dfdu[0, 0] = 1.0 / self.mass
        dfdu[1, 1] = 1.0 / self.mass
        dfdx[0, 2] = -2.0 * self.c_drag * x[2]
        dfdx[1, 3] = -2.0 * self.c_drag * x[3]
        return dfdx, dfdu


class DifferentialActionModelLQ(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1e-2, isTerminal=False):
        self.dynamics = PointMassDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, self.dynamics.nu, self.dynamics.ndx
        )
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.mass = self.dynamics.mass
        self.dt = dt

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return cost

    def _terminal_cost(self, x, u):
        cost = (
            200 * ((x[0] - 1.0) ** 2)
            + 200 * (x[1] ** 2)
            + 10 * (x[2] ** 2)
            + 10 * (x[3] ** 2)
        )
        return cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xout = np.zeros(self.state.nv)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = self.dynamics.nonlinear_dynamics(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([2, 4])
        Fu = np.zeros([2, 2])

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = 400.0 * (x[0] - 1)
            Lx[1] = 400.0 * x[1]
            Lx[2] = 20.0 * x[2]
            Lx[3] = 20.0 * x[3]
            Lxx[0, 0] = 400.0
            Lxx[1, 1] = 400.0
            Lxx[2, 2] = 20.0
            Lxx[3, 3] = 20.0
        else:
            Lx[0] = 2.0 * (x[0] - 1)
            Lx[1] = 2.0 * x[1]
            Lx[2] = 2.0 * x[2]
            Lx[3] = 2.0 * x[3]
            Lu[0] = 2 * u[0]
            Lu[1] = 2 * u[1]
            Lxx[0, 0] = 2
            Lxx[1, 1] = 2
            Lxx[2, 2] = 2
            Lxx[3, 3] = 2
            Luu[0, 0] = 2.0
            Luu[1, 1] = 2

            Fu[0, 0] = 1.0 / self.mass
            Fu[1, 1] = 1.0 / self.mass
            Fx[0, 2] = -2.0 * self.dynamics.c_drag * x[2]
            Fx[1, 3] = -2.0 * self.dynamics.c_drag * x[3]

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()



class IntegratedActionModelLQ(crocoddyl.IntegratedActionModelEuler): 
    def __init__(self, diffModel, dt=1.e-2):
        super().__init__(diffModel, dt)
        self.diffModel = diffModel 
        self.intModel = crocoddyl.IntegratedActionModelEuler(self.diffModel, dt) 
        self.Fxx = np.zeros([self.state.ndx, self.state.ndx, self.state.ndx])
        self.Fxu = np.zeros([self.state.ndx, self.state.ndx, self.nu])
        self.Fuu = np.zeros([self.state.ndx, self.nu, self.nu])
    
    def calcFxx(self, x, u): 
        self.Fxx[0, 2, 2] = -2 * self.diffModel.dynamics.c_drag * self.dt ** 2
        self.Fxx[1, 3, 3] = -2 * self.diffModel.dynamics.c_drag * self.dt ** 2
        self.Fxx[2, 2, 2] = -2 * self.diffModel.dynamics.c_drag * self.dt
        self.Fxx[3, 3, 3] = -2 * self.diffModel.dynamics.c_drag * self.dt


    def calc(self, data, x, u=None):
        if u is None:
            self.intModel.calc(data, x)
        else:
            self.intModel.calc(data, x, u)
        
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.intModel.calcDiff(data, x)
            u = np.zeros(self.nu)
        else:
            self.intModel.calcDiff(data, x, u)
        self.calcFxx(x,u)
        






if __name__ == "__main__":
    print(" Testing with DDP ".center(LINE_WIDTH, "#"))
    lq_diff_running = DifferentialActionModelLQ()
    lq_diff_terminal = DifferentialActionModelLQ(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))
    dt = 0.01
    horizon = 300
    x0 = np.zeros(4)
    lq_running = IntegratedActionModelLQ(lq_diff_running, dt)
    lq_terminal = IntegratedActionModelLQ(lq_diff_terminal, dt)
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    ddp = crocoddyl.SolverGNMS(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (horizon + 1)
    us = [np.zeros(2)] * horizon
    converged = ddp.solve(xs, us)


    plt.figure("trajectory plot")
    plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("DDP")
    plt.show()