import gym
from gym import spaces
import numpy as np
import math


class SoccerBoxEnv(gym.Env):
    """
        A tries to move ball to left goal, B tries to move ball to right goal
        Actions [0 : Left, 1 : Up, 2 : Right, 3 : Down, 4 : Stand]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, board_height=4, board_width=5,
                 goal_upper_bound=2, goal_lower_bound=1,
                 draw_probability=0.01,
                 initial_position_A=None, initial_position_B=None,
                 initial_ball_owner=None):
        super(SoccerBoxEnv, self).__init__()

        self.board_height = board_height
        self.board_width = board_width
        self.goal_upper_bound = goal_upper_bound
        self.goal_lower_bound = goal_lower_bound
        self.draw_probability = draw_probability
        self.initial_position_A = initial_position_A
        self.initial_position_B = initial_position_B
        self.initial_ball_owner = initial_ball_owner

        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0]), high=np.array([1,1,1,1,1]))
        self.action_space = spaces.Discrete(5)
        self.bction_space = spaces.Discrete(5)

        self.position_A = self.initial_position_A
        self.position_B = self.initial_position_B
        self.ball_owner = self.initial_ball_owner


    def seed(self, seed=None):
        np.random.seed(seed)


    def step(self, action, bction=None):
        if np.random.rand() < self.draw_probability:
            obs = self.state_to_observation()
            return obs, 0, True, 'draw'
        else:
            first = self.choose_turn()
            actions = [action, bction]
            rew, done = self.player_move(first, actions[first])
            if done:
                obs = self.state_to_observation()
                info = {'winner': 'A' if rew > 0 else 'B'}
                return obs, rew, done, info
            else:
                rew, done = self.player_move(1 - first, actions[1 - first])
                obs = self.state_to_observation()
                info = {'winner': 'A' if rew > 0 else 'B'} if done else None
                return obs, rew, done, info


    def reset(self, position_A=None, position_B=None, ball_owner=None):
        if position_A is None:
            position_A = self.initial_position_A
        if position_B is None:
            position_B = self.initial_position_B
        if ball_owner is None:
            ball_owner = self.initial_ball_owner

        if position_A is not None and position_B is not None and ball_owner is not None:
            self.position_A = position_A
            self.position_B = position_B
            self.ball_owner = ball_owner
            return self.state_to_observation()
        else:
            xA = np.random.randint(self.board_width)
            yA = np.random.randint(self.board_height)
            xB, yB = xA, yA
            while xB == xA and yB == yA:
                xB = np.random.randint(self.board_width)
                yB = np.random.randint(self.board_height)
            ball_owner = bool(np.random.randint(2))
            self.position_A = np.array([xA,yA])
            self.position_B = np.array([xB,yB])
            self.ball_owner = ball_owner
            obs = self.state_to_observation()
            return obs


    def state_to_observation(self):
        xA, yA = self.position_A
        xB, yB = self.position_B
        obs = np.array(
            [[xA / self.board_width,
              yA / self.board_height,
              xB / self.board_width,
              yB / self.board_height,
              self.ball_owner]]
        )
        return obs


    def choose_turn(self):
        return np.random.randint(0, 2)

    def player_move(self, player, action):
        switcher = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [0, 0],
        }

        if player == 0:
            player_position = self.position_A
            opponent_position = self.position_B
        elif player == 1:
            player_position = self.position_B
            opponent_position = self.position_A
        else:
            raise NotImplementedError

        new_player_position = np.array(player_position) + np.array(switcher.get(action))
        # If it's opponent position
        if np.all(new_player_position == opponent_position):
            self.ball_owner = 1 - player
            return 0, False
        # If it's the goal
        elif self.ball_owner == player and self.is_A_goal(new_player_position):
            return 1, True
        elif self.ball_owner == player and self.is_B_goal(new_player_position):
            return -1, True
        # If it's in board
        elif self.is_board(new_player_position):
            if player == 0:
                self.position_A = new_player_position
            elif player == 1:
                self.position_B = new_player_position
            else:
                raise NotImplementedError
            return 0, False
        else:
            return 0, False

    def is_A_goal(self, position):
        if position[0] < 0 and \
                (self.goal_lower_bound <= position[1] <= self.goal_upper_bound):
            return True
        else:
            return False

    def is_B_goal(self, position):
        if position[0] >= self.board_width and \
                (self.goal_lower_bound <= position[1] <= self.goal_upper_bound):
            return True
        else:
            return False

    def is_board(self, position):
        return (0 <= position[0] < self.board_width and
                0 <= position[1] < self.board_height)


    def mirror_observation(self):
        xB, yB = self.board_width-1-self.position_A[0], self.position_A[1]
        xA, yA = self.board_width-1-self.position_B[0], self.position_B[1]
        ball_owner = 1 - self.ball_owner

        obs = np.array(
            [[xA / self.board_width,
              yA / self.board_height,
              xB / self.board_width,
              yB / self.board_height,
              ball_owner]]
        )
        return obs

    def mirror_action(self, act):
        if act == 0:
            act = 2
        elif act == 2:
            act = 0
        return act


    def render(self, mode='human', close=False):
        board = ''
        for y in range(self.board_height)[::-1]:
            for x in range(self.board_width):
                if np.all(self.position_A == [x, y]):
                    board += 'A' if self.ball_owner == 0 else 'a'
                elif np.all(self.position_B == [x, y]):
                    board += 'B' if self.ball_owner == 1 else 'b'
                else:
                    board += '-'
            board += '\n'
        print(board)