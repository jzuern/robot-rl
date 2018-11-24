import gym
from gym import spaces
import numpy as np
from gym import utils
from random import randint


class Obstacle:

    def __init__(self):
        self.hole_top = randint(0, 30)
        self.hole_bottom = self.hole_top + 10
        self.pos_x = 40

    def reset(self):
        self.hole_top = randint(0, 30)
        self.hole_bottom = self.hole_top + 10
        self.pos_x = 40

    def step(self):
        self.pos_x -= 1  # increment position
        if self.pos_x < 0:  # reset obstacle if outside environment
            self.reset()

    def set_pos_x(self, pos_x):
        self.pos_x = pos_x

    def get_pos(self):
        return self.pos_x

    def get_hole(self):
        return self.hole_top, self.hole_bottom


class Robot:

    def __init__(self):
        self.height = 0

    def move(self, direction):
        if direction == 0 and self.height > 0:
            self.height -= 2  # move up
        if direction == 1 and self.height < 40-5:
            self.height += 2  # move down
        if direction == 2:
            self.height = self.height  # stay

    def set_height(self, height):
        self.height = height

    def get_height(self):
        return self.height

    def get_x(self):
        return 20

    def reset(self):
        self.height = randint(5, 35)


class RoadEnv(gym.Env, utils.EzPickle):

    def __init__(self):

        from gym.envs.classic_control import rendering

        self.viewer = rendering.SimpleImageViewer()

        self._action_set = {0, 1}  # go up, go down
        self.action_space = spaces.Discrete(len(self._action_set))

        # init obstacle
        self.obstacle = Obstacle()

        # init robot
        self.robby = Robot()

    # if game is over, it resets itself
    def reset_game(self):

        self.robby.reset()
        self.obstacle.reset()

    # a single time step in the environment
    def step(self, a):

        reward, game_over = self.act(a)
        ob = self._get_obs()
        info = {}

        return ob, reward, game_over, info

    # perform action a
    def act(self, a):

        self.obstacle.step()
        self.robby.move(a)

        rob_pos_y = self.robby.get_height()
        rob_pos_x = self.robby.get_x()

        top, bottom = self.obstacle.get_hole()
        obstacle_pos_x = self.obstacle.get_pos()

        distance_x = abs(rob_pos_x - obstacle_pos_x)

        collide_x = distance_x < 5
        collide_y = rob_pos_y < top or (rob_pos_y + 5 > bottom)

        game_over = False
        reward = 0.0

        if collide_x and collide_y:
            game_over = True
        else:
            reward = 0.1

        return reward, game_over

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        self.reset_game()
        return self._get_obs()

    def _get_obs(self):
        img = self._get_image

        # image must be expanded along first dimension for keras
        return np.expand_dims(img, axis=0)

    def render(self):

        img = self._get_image

        # image must be expanded to 3 color channels to properly show the content
        img = np.repeat(img, 3, axis=2)

        # show frame on display
        self.viewer.imshow(img)

        return self.viewer.isopen

    @property
    def _get_image(self):

        img = np.zeros(shape=(40, 40, 1), dtype=np.uint8)

        obstacle_x = self.obstacle.get_pos()
        width = 4

        img[:, obstacle_x:obstacle_x + width, 0] = 128

        top, bottom = self.obstacle.get_hole()

        img[top:bottom, obstacle_x:obstacle_x + width, 0] = 0

        rob_y = self.robby.get_height()
        rob_x = self.robby.get_x()

        img[rob_y:rob_y + width, rob_x:rob_x + width, 0] = 255

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
