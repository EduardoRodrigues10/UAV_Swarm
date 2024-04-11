from QLearning import QLearning
import time
import sys
import numpy as np
import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.high_level_commander import HighLevelCommander

FORWARD, BACKWARD, LEFT, RIGHT = range(4)

action_duration = 1.0


def execute_action(commander, action):
    if action == FORWARD:
        print("FORWARD")
        commander.go_to(0.3, 0.0, 0.0, 0.0, action_duration, relative=True)
        time.sleep(action_duration)
    elif action == BACKWARD:
        print("BACKWARD")
        commander.go_to(-0.3, 0.0, 0.0, 0.0, action_duration, relative=True)
        time.sleep(action_duration)
    elif action == LEFT:
        print("LEFT")
        commander.go_to(0.0, -0.3, 0.0, 0.0, action_duration, relative=True)
        time.sleep(action_duration)
    elif action == RIGHT:
        print("RIGHT")
        commander.go_to(0.0, 0.3, 0.0, 0.0, action_duration, relative=True)
        time.sleep(action_duration)


def activate_high_level_commander(scf):
    scf.cf.param.set_value('commander.enHighLevel', '1')


def main(scf, params):
    cf = scf.cf
    commander = HighLevelCommander(cf)

    commander.takeoff(0.5, action_duration)
    time.sleep(3.0)

    for action in actions:
        execute_action(commander, action)
        time.sleep(1.0)

    time.sleep(1.0)
    commander.land(0.0, action_duration)
    time.sleep(2.0)
    commander.stop()


uris = {
    'radio://0/20/2M/E7E7E7E701',
    'radio://0/20/2M/E7E7E7E703',
}

params = {
    'radio://0/20/2M/E7E7E7E701': [{'d': 1}],
    'radio://0/20/2M/E7E7E7E703': [{'d': 3}],
}

actions = None

if __name__ == "__main__":

    nb_rows, nb_cols = 3, 3
    grid1 = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])

    grid2 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

    grid3 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ])
    grids = [grid1, grid2, grid3]

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 10000

    qlearning = QLearning(nb_rows, nb_cols, alpha, gamma, epsilon)
    qlearning.train(episodes, grids)
    actions = qlearning.get_actions(grids)

    print("Q-Learning OK")

    for action in actions:
        print(action)
    '''
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(activate_high_level_commander)
        # swarm.parallel_safe(reset_estimator)
        swarm.parallel_safe(main, args_dict=params)
    '''