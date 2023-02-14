import numpy as np
from numpy import linalg

from LQproblem import *
import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverGNMS

LINE_WIDTH = 100 

VERBOSE = False    

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class GNMS_linesearch(SolverGNMS):
    def __init__(self, shootingProblem):
        SolverGNMS.__init__(self, shootingProblem)
        # self.ddp = crocoddyl.SolverGNMS(shootingProblem)
        self.ln_max = 10
        self.mu = 1e10

    def merit(self):
        return self.cost + self.mu * self.compute_gaps()
        
    def merit_try(self):
        return self.cost_try + self.mu * np.linalg.norm(list(self.fs_try), 1)

    def compute_gaps(self):
        return np.linalg.norm(list(self.fs), 1)

    def solve(self, xs, us, maxiter=20, isFeasible=False):
        xs_ = list(self.xs)
        us_ = list(self.us)
        self.setCandidate(xs, us, False)
        for i in range(maxiter):
            self.calcDiff()
            self.backwardPass()
            self.cost_try = np.inf
            set_line_step = 1
            it = 0 

            self.forwardPass(set_line_step)
            while self.merit_try() > self.merit() and it > self.ln_max:
                set_line_step *= 0.5
                self.forwardPass(set_line_step)
                # print(self.merit_try(), self.merit(), set_line_step)
                it += 1

            print("iteration", i, "cost", self.cost, "gaps", self.compute_gaps(), "merit", self.merit())

            if abs(self.merit_try() - self.merit() ) < 1e-6 or self.ln_max == it:
                print("Breaking no improvement")
                break

            self.setCandidate(self.xs_try, self.us_try, False) 

    # def allocateData(self):

    #     self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
    #     self.xs_try[0][:] = self.problem.x0.copy()
    #     self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 




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

    ddp = GNMS_linesearch(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (horizon + 1)
    us = [1000*np.ones(2)] * horizon
    converged = ddp.solve(xs, us)
    print("final", ddp.cost)

    plt.figure("trajectory plot")
    plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("DDP")
    plt.show()