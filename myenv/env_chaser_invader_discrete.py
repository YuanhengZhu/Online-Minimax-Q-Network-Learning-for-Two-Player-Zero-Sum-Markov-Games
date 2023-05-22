import gym
from gym import spaces
import numpy as np
import math


class ChaserInvaderDiscreteEnv(gym.Env):
    """
    invader tries to reach the centre goal, while chaser wants to block it as far as possible
    Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, board_bottom=-5, board_top=5, board_left=-5, board_right=5,
                 initial_chaser_position=None, initial_invader_position=None,
                 draw_probability=0.01):
        super(ChaserInvaderDiscreteEnv, self).__init__()

        self.board_bottom = board_bottom
        self.board_top = board_top
        self.board_left = board_left
        self.board_right = board_right
        self.initial_chaser_position = initial_chaser_position
        self.initial_invader_position = initial_invader_position
        self.draw_probability = draw_probability

        self.observation_space = spaces.Discrete(
            ((self.board_right - self.board_left + 1) * (
                    self.board_top - self.board_bottom + 1))**2
            )
        self.action_space = spaces.Discrete(5)
        self.bction_space = spaces.Discrete(5)

        self.chaser_position = self.initial_chaser_position
        self.invader_position = self.initial_invader_position


    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action, bction=None):
        if np.random.rand() < self.draw_probability:
            obs = self.state_to_observation()
            return obs, 0, True, 'draw'
        else:
            switcher = {
                0: [-1, 0],
                1: [0, 1],
                2: [1, 0],
                3: [0, -1],
                4: [0, 0],
            }
            new_chaser_position = np.array(self.chaser_position) + np.array(
                switcher.get(action))
            if self.is_board(new_chaser_position):
                self.chaser_position = new_chaser_position

            new_invader_position = np.array(self.invader_position) + np.array(
                switcher.get(bction))
            if self.is_board(new_invader_position):
                self.invader_position = new_invader_position

            obs = self.state_to_observation()
            if self.is_invader_goal():
                info = {'winner': 'invader'}
                return obs, -10, True, info
            elif self.is_chaser_catch():
                info = {'winner': 'chaser'}
                rew = np.sum(np.absolute(self.invader_position))
                return obs, rew, True, info
            else:
                return obs, 0, False, None


    def reset(self, chaser_position=None, invader_position=None):
        if chaser_position is None:
            chaser_position = self.initial_chaser_position
        if invader_position is None:
            invader_position = self.initial_invader_position

        if chaser_position is not None and invader_position is not None:
            self.chaser_position = chaser_position
            self.invader_position = invader_position
        else:
            self.chaser_position = np.array(
                [np.random.random_integers(self.board_left, self.board_right),
                 np.random.random_integers(self.board_bottom, self.board_top)])
            self.invader_position = np.array(
                [np.random.random_integers(self.board_left, self.board_right),
                 np.random.random_integers(self.board_bottom, self.board_top)])

        return self.state_to_observation()


    def probability_model(self, action, bction):
        # pure model won't change env state, but to see how many kinds and probabilities of next dynamics
        probs, obses_tp1, rews, dones = [], [], [], []
        tmp_chaser_position = self.chaser_position
        tmp_invader_position = self.invader_position

        new_obs, rew, done, info = self.step(action, bction)
        probs.append(1.0)
        obses_tp1.append(new_obs)
        rews.append(rew)
        dones.append(done)

        self.reset(tmp_chaser_position, tmp_invader_position)
        return probs, obses_tp1, rews, dones


    def state_to_observation(self):
        width = self.board_right - self.board_left + 1
        height = self.board_top - self.board_bottom + 1
        xA, yA = self.chaser_position - np.array([self.board_left, self.board_bottom])
        xB, yB = self.invader_position - np.array([self.board_left, self.board_bottom])
        sA = yA * width + xA
        sB = yB * width + xB
        obs = sA * (width * height) + sB
        return obs

    def observation_to_state(self, obs):
        width = self.board_right - self.board_left + 1
        height = self.board_top - self.board_bottom + 1
        sA = math.floor(obs / (width * height))
        sB = obs % (width * height)

        xA = sA % width
        yA = math.floor(sA / width)
        xB = sB % width
        yB = math.floor(sB / width)

        self.chaser_position = np.array([xA, yA])
        self.invader_position = np.array([xB, yB])


    def is_invader_goal(self):
        return np.all(self.invader_position == [0,0])

    def is_chaser_catch(self):
        return (abs(self.chaser_position[0] - self.invader_position[0]) <= 1 and
                abs(self.chaser_position[1] - self.invader_position[1]) <= 1)

    def is_board(self, position):
        return (self.board_left <= position[0] <= self.board_right
                and self.board_bottom <= position[1] <= self.board_top)

    def render(self, mode='human', close=False):
        board = ''
        for y in range(self.board_bottom, self.board_top + 1)[::-1]:
            for x in range(self.board_left, self.board_right + 1):
                if np.all(np.array([x, y]) == self.chaser_position):
                    if np.all(self.chaser_position == self.invader_position):
                        c = '*'
                    else:
                        c = 'C'
                elif np.all(np.array([x, y]) == self.invader_position):
                    if np.all(self.invader_position == [0,0]):
                        c = 'o'
                    else:
                        c = 'I'
                elif np.all(np.array([x, y]) == [0,0]):
                    c = 'G'
                else:
                    c = '-'
                board += c
            board += '\n'

        print(board)