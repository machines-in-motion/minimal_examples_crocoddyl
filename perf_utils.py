
import time
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from core_mpc import misc_utils 
logger = misc_utils.CustomLogger(__name__, misc_utils.GLOBAL_LOG_LEVEL, misc_utils.GLOBAL_LOG_FORMAT).logger
import crocoddyl


class MPCBenchmark:
    def __init__(self):
        self.solving_times = []
        self.solving_times_per_iter = []
        self.number_of_iterations = []
        self._start_time = None
        
        self.profiles = []

        # self.profiles.append('SolverDDP::Qu')
        # self.profiles.append('SolverDDP::Quu')
        # self.profiles.append('SolverDDP::Quu_inv')
        # self.profiles.append('SolverDDP::Qx')
        # self.profiles.append('SolverDDP::Qxu')
        # self.profiles.append('SolverDDP::Qxx')
        # self.profiles.append('SolverDDP::Vxx')
        # self.profiles.append('SolverDDP::computeGains')

        # self.profiles.append('ShootingProblem::calc')
        # self.profiles.append('ShootingProblem::calcDiff')
        # self.profiles.append('SolverGNMS::backwardPass')
        # self.profiles.append('SolverGNMS::calcDiff')
        self.profiles.append('SolverGNMS::computeDirection')
        self.profiles.append('SolverGNMS::tryStep')
        self.profiles.append('SolverGNMS::forwardPass')
        self.profiles.append('SolverGNMS::solve')

        self.avg = {}
        self.min = {}
        self.max = {}
        self.tot = {}
        for profile in self.profiles:
            self.avg[profile] = []
            self.min[profile] = []
            self.max[profile] = []
            self.tot[profile] = []

    def start_croco_profiler(self):
        crocoddyl.stop_watch_reset_all()
        crocoddyl.enable_profiler()

    def stop_croco_profiler(self):
        crocoddyl.disable_profiler()

    def print_croco_profiler(self):
        crocoddyl.stop_watch_report(3)

    def record_profiles(self):
        for profile in self.profiles:
            # logger.debug(profile)
            self.avg[profile].append(1e3*crocoddyl.stop_watch_get_average_time(profile))
            self.min[profile].append(1e3*crocoddyl.stop_watch_get_min_time(profile))
            self.max[profile].append(1e3*crocoddyl.stop_watch_get_max_time(profile))
            self.tot[profile].append(1e3*crocoddyl.stop_watch_get_total_time(profile))
        
        # self.print_croco_profiler()

    def start_timer(self):
        self._start_time = 1e-6*time.perf_counter_ns()

    def stop_timer(self, nb_iter=None):
        elapsed_time = 1e-6*time.perf_counter_ns() - self._start_time
        self.solving_times.append(elapsed_time)
        if(nb_iter is not None and nb_iter > 0):
            self.solving_times_per_iter.append(elapsed_time/float(nb_iter))
            self.number_of_iterations.append(nb_iter)
        self._start_time = None

    def plot_profile(self, profile, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True):
        '''
        Plot 1 profile
        Input:
            profile                   : profile name
            SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
            SHOW                      : show plots
        '''
        logger.info('Plotting MPC profiles...')
        N = len(self.avg[profile])
        fig, ax = plt.subplots(1, 1, figsize=(19.2,10.8), sharex='col') 
        # avg
        tspan = np.linspace(0,N-1, N)
        ax.plot(tspan, self.avg[profile], color='b', linestyle='-', marker='o', markerfacecolor='b', markersize=9, label=profile+' time')
        ax.fill_between(tspan, self.min[profile], self.max[profile], alpha=.5, linewidth=0)
        ax.set_xlabel('MPC cycle', fontsize=20)
        ax.set_ylabel('time (ms)', fontsize=18)
        ax.grid(True)
        # import pdb
        # pdb.set_trace()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 18})
        fig.suptitle(profile + ' timings', size=24)
        if(SAVE):
            figs = {'bench': fig}
            if(SAVE_DIR is None):
                logger.error("Please specify SAVE_DIR")
            if(SAVE_NAME is None):
                SAVE_NAME = 'testfig'
            for name, fig in figs.items():
                fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
        if(SHOW):
            plt.show()       
        return fig

    def plot_profiles(self, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True):
        '''
        Plot 1 profile
        Input:
            SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
            SHOW                      : show plots
        '''
        logger.info('Plotting MPC profiles...')
        nprofiles = len(self.profiles)
        fig, ax = plt.subplots(1, 1, figsize=(19.2,10.8), sharex='col') 
        colormap = plt.cm.get_cmap('nipy_spectral', nprofiles)
        for i in range(nprofiles):
            profile = self.profiles[i]
            col = colormap(i)
            N = len(self.avg[profile])
            tspan = np.linspace(0,N-1, N)
            ax.plot(tspan, self.avg[profile], color=col, linestyle='-', marker='o', markerfacecolor=col, markersize=9, label=profile)
            ax.fill_between(tspan, self.min[profile], self.max[profile], alpha=0.3, linewidth=0, color=col)
            # ax[i].set_xlabel('MPC cycle', fontsize=20)
        ax.set_ylabel('time (ms)', fontsize=18)
        ax.set_yscale('log')
        ax.grid(True)
            # import pdb
            # pdb.set_trace()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 18})
        fig.suptitle('Timings', size=24)
        if(SAVE):
            figs = {'bench': fig}
            if(SAVE_DIR is None):
                logger.error("Please specify SAVE_DIR")
            if(SAVE_NAME is None):
                SAVE_NAME = 'testfig'
            for name, fig in figs.items():
                fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
        if(SHOW):
            plt.show()       
        return fig

    def plot_avg_profiles(self, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True):
        '''
        Plot 1 profile
        Input:
            SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
            SHOW                      : show plots
        '''
        logger.info('Plotting MPC profiles...')
        profiles = self.profiles # remove profiles here if necessary

        nprofiles = len(profiles) 
        timings = [sum(self.avg[profile])/len(self.avg[profile]) for profile in profiles]
        fig, ax = plt.subplots(1, 1, figsize=(19.2,10.8), sharex='col') 
        colormap = plt.cm.get_cmap('hsv', nprofiles)
        ax.bar(profiles, timings, color=[colormap(i) for i in range(nprofiles)])

        # colormap = plt.cm.get_cmap('hsv', nprofiles)
        # # print(colormap)
        # for i in range(nprofiles):
        #     # color = lambda : [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
        #     profile = profiles[i]
        #     if('solve' not in profile):
        #         logger.debug(str(colormap(i)))
        #         # print(color)
        #         N = len(self.avg[profile])
        #         tspan = np.linspace(0,N-1, N)
        #         ax.plot(tspan, self.avg[profile], color=colormap(i), linestyle='-', marker='o', markerfacecolor=colormap(i), markersize=9, label=profile+' time')
        #         ax.fill_between(tspan, self.min[profile], self.max[profile], alpha=0.3, linewidth=0)
        #         # ax[i].set_xlabel('MPC cycle', fontsize=20)
        ax.set_ylabel('time (ms)', fontsize=18)
        # ax.set_yscale('log')
        ax.grid(True)
            # import pdb
            # pdb.set_trace()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', prop={'size': 18})
        fig.suptitle('Timings', size=24)
        if(SAVE):
            figs = {'bench': fig}
            if(SAVE_DIR is None):
                logger.error("Please specify SAVE_DIR")
            if(SAVE_NAME is None):
                SAVE_NAME = 'testfig'
            for name, fig in figs.items():
                fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
        if(SHOW):
            plt.show()       
        return fig

    def plot_timer(self, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True):
        '''
        Plot MPC solving times
        Input:
            SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
            SHOW                      : show plots
        '''
        logger.info('Plotting MPC benchmark...')
        N = len(self.solving_times)
        fig, ax = plt.subplots(3, 1, figsize=(19.2,10.8), sharex='col') 
        ax[0].plot(np.linspace(0,N-1, N), self.solving_times, color='r', linestyle='-', marker='o', markerfacecolor='r', markersize=9, label='Total time')
        # ax[0].plot(np.linspace(0,N-1, N), self.tot['SolverFDDP::solve'], color='b', linestyle='-', marker='o', markerfacecolor='b', markersize=9)
        # ax[0].set_xlabel('MPC cycle', fontsize=20)
        ax[0].set_ylabel('Tot. time (ms)', fontsize=18)
        ax[0].grid(True)
        # import pdb
        # pdb.set_trace()
        N = len(self.solving_times_per_iter)
        ax[1].plot(np.linspace(0,N-1, N), self.solving_times_per_iter, color='b', linestyle='-', marker='o', markerfacecolor='b', markersize=9, label='Avg time per iteration')
        # ax[1].plot(np.linspace(0,N-1, N), self.tot['SolverFDDP::solve']/self.number_of_iterations, color='r', linestyle='-', marker='o', markerfacecolor='r', markersize=9)
        # ax[1].set_xlabel('MPC cycle', fontsize=20)
        ax[1].set_ylabel('Avg time/it (ms)', fontsize=18)
        ax[1].grid(True)
        N = len(self.number_of_iterations)
        ax[2].plot(np.linspace(0,N-1, N), self.number_of_iterations,color='g', linestyle='-', marker='o', markerfacecolor='g', markersize=9, label='Number of iterations')
        ax[2].set_xlabel('MPC cycle', fontsize=18)
        ax[2].set_ylabel('# iters', fontsize=18)
        # ax[2].yaxis.set_major_locator(plt.MaxNLocator(4))
        # ax[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%1i'))
        ax[2].grid(True)

        handles0, labels0 = ax[0].get_legend_handles_labels()
        handles1, labels1 = ax[1].get_legend_handles_labels()
        handles2, labels2 = ax[2].get_legend_handles_labels()
        handles = handles0 + handles1 + handles2
        labels = labels0 + labels1 + labels2
        fig.legend(handles, labels, loc='upper right', prop={'size': 18})
        # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        fig.align_ylabels(ax[:])
        fig.suptitle('MPC performances', size=24)
        if(SAVE):
            figs = {'bench': fig}
            if(SAVE_DIR is None):
                logger.error("Please specify SAVE_DIR")
            if(SAVE_NAME is None):
                SAVE_NAME = 'testfig'
            for name, fig in figs.items():
                fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
        if(SHOW):
            plt.show()       
        return fig