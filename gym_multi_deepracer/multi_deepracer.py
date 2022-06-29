from calendar import c
import Box2D, gym
import gym.envs.box2d.car_dynamics as car_dynamics
from gym.utils import seeding, EzPickle
from gym import spaces
import numpy as np
import math

from shapely.geometry import Point

from src.friction_detector import FrictionDetector
from src.deepracer_parameters import DeepRacerParameters
from src.track_builder import TrackBuilder
from src.gym_renderer import GymRenderer

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Gianluigi Mucciolo. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

TRACK_SCALE = 6.0
TRACK_RADIUS = 900 / TRACK_SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / TRACK_SCALE  # Game over boundary
FPS = 50
ZOOM = 2.7  # Camera zoom

# Set to False for fixed view (don't use zoom)
ZOOM_FOLLOW = True

TRACK_DETAIL_STEP = 21 / TRACK_SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / TRACK_SCALE
BORDER = 8 / TRACK_SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

# Specify different car colors
CAR_COLORS = [
    (0.8, 0.0, 0.0),
    (0.0, 0.0, 0.8),
    (0.0, 0.8, 0.0),
    (0.0, 0.8, 0.8),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 0.0),
    (0.8, 0.0, 0.8),
    (0.8, 0.8, 0.0),
]

# Distance between cars
LINE_SPACING = 5  # Starting distance between each pair of cars
LATERAL_SPACING = 3  # Starting side distance between pairs of cars

# Penalizing backwards driving
BACKWARD_THRESHOLD = np.pi / 2
K_BACKWARD = 0  # Penalty weight: backwards_penalty = K_BACKWARD * angle_diff  (if angle_diff > BACKWARD_THRESHOLD)


class MultiDeepRacer(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(
        self,
        num_agents=1,
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        direction="CCW",
        verbose=1,
        use_ego_color=True,
    ):
        super(MultiDeepRacer, self).__init__()
        EzPickle.__init__(self)
        self.seed()

        self.new_lap = False
        self.num_agents = num_agents
        self.contactListener_keepref = FrictionDetector(
            self, lap_complete_percent=0.95, road_color=ROAD_COLOR
        )
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = [None] * num_agents
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars = [None] * num_agents
        self.car_order = None  # Determines starting positions of cars
        self.reward = np.zeros(num_agents)
        self.prev_reward = np.zeros(num_agents)
        self.tile_visited_count = [0] * num_agents
        self.verbose = verbose
        self.driving_backward = np.zeros(num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(num_agents, dtype=bool)
        self.use_random_direction = (
            use_random_direction  # Whether to select direction randomly
        )
        self.episode_direction = direction  # Choose 'CCW' (default) or 'CW' (flipped)
        if self.use_random_direction:  # Choose direction randomly
            self.episode_direction = np.random.choice(["CW", "CCW"])
        self.backwards_flag = (
            backwards_flag  # Boolean for rendering backwards driving flag
        )
        self.h_ratio = (
            h_ratio  # Configures vertical location of car within rendered window
        )
        self.use_ego_color = (
            use_ego_color  # Whether to make ego car always render as the same color
        )

        self.action_lb = np.tile(np.array([-1, +0, +0]), 1)  # self.num_agents)
        self.action_ub = np.tile(np.array([+1, +1, +1]), 1)  # self.num_agents)

        self.action_space = spaces.Box(
            self.action_lb, self.action_ub
        )  # (steer, gas, brake) x N
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3)
        )

        self.renderer = GymRenderer(self)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []

        for car in self.cars:
            car.destroy()

    def _create_track(self):
        (self.track, self.road_poly, self.road, self.road_poly_shapely,) = TrackBuilder(
            self.np_random, ROAD_COLOR, self.world, self.num_agents
        ).get()
        return True

    def reset_car(self, car_id):
        (angle, pos_x, pos_y) = self.track[0][1:4]

        # Specify line and lateral separation between cars
        line_spacing = LINE_SPACING
        lateral_spacing = LATERAL_SPACING

        # index into positions using modulo and pairs
        line_number = math.floor(self.car_order[car_id] / 2)  # Starts at 0
        side = (2 * (self.car_order[car_id] % 2)) - 1  # either {-1, 1}

        # Compute offsets from start (this should be zero for first pair of cars)
        dx = self.track[-line_number * line_spacing][2] - pos_x  # x offset
        dy = self.track[-line_number * line_spacing][3] - pos_y  # y offset

        # Compute angle based off of track index for car
        angle = self.track[-line_number * line_spacing][1]
        if self.episode_direction == "CW":  # CW direction indicates reversed
            angle -= np.pi  # Flip direction is either 0 or pi

        # Compute offset angle (normal to angle of track)
        norm_theta = angle - np.pi / 2

        # Compute offsets from position of original starting line
        new_x = pos_x + dx + (lateral_spacing * np.sin(norm_theta) * side)
        new_y = pos_y + dy + (lateral_spacing * np.cos(norm_theta) * side)

        # Display spawn locations of cars.
        # print(f"Spawning car {car_id} at {new_x:.0f}x{new_y:.0f} with "
        #       f"orientation {angle}")

        # Create car at location with given angle
        self.cars[car_id] = car_dynamics.Car(self.world, angle, new_x, new_y)
        self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

        # This will be used to identify the car that touches a particular tile.
        for wheel in self.cars[car_id].wheels:
            wheel.car_id = car_id

        print(self.reward, self.prev_reward)
        self.reward[car_id] = 0.0
        self.prev_reward[car_id] = 0.0
        print(self.reward, self.prev_reward)

    def reset(self):
        self._destroy()
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.tile_visited_count = [0] * self.num_agents
        self.t = 0.0
        self.steps = 0.0
        self.road_poly = []

        # Reset driving backwards/on-grass states and track direction
        self.driving_backward = np.zeros(self.num_agents, dtype=bool)
        self.driving_on_grass = np.zeros(self.num_agents, dtype=bool)
        if self.use_random_direction:  # Choose direction randomly
            self.episode_direction = np.random.choice(["CW", "CCW"])

        # Set positions of cars randomly
        ids = [i for i in range(self.num_agents)]
        shuffle_ids = np.random.choice(ids, size=self.num_agents, replace=False)
        self.car_order = {i: shuffle_ids[i] for i in range(self.num_agents)}

        success = False
        while not success:
            success = self._create_track()
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many of this messages)"
                )

        for car_id in range(self.num_agents):
            self.reset_car(car_id=car_id)
        return self.step(None)[0]

    def step(self, action):
        """ Run environment for one timestep. 
        
        Parameters:
            action(np.ndarray): Numpy array of shape (num_agents,3) containing the
                commands for each car. Each command is of the shape (steer, gas, brake).
        """

        if action is not None:
            # NOTE: re-shape action as input action is flattened
            action = np.reshape(action, (self.num_agents, -1))
            for car_id, car in enumerate(self.cars):
                car.steer(-action[car_id][0])
                car.gas(action[car_id][1])
                car.brake(action[car_id][2])

        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.steps += 1.0

        self.state = self.render("state_pixels")
        step_reward = np.zeros(self.num_agents)

        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER

            # NOTE(IG): Probably not relevant. Seems not to be used anywhere. Commented it out.
            # self.cars[0].fuel_spent = 0.0

            step_reward = self.reward - self.prev_reward

            # Add penalty for driving backward
            for car_id, car in enumerate(self.cars):  # Enumerate through cars

                # Get car speed
                vel = car.hull.linearVelocity
                if np.linalg.norm(vel) > 0.5:  # If fast, compute angle with v
                    car_angle = -math.atan2(vel[0], vel[1])
                else:  # If slow, compute with hull
                    car_angle = car.hull.angle

                # Map angle to [0, 2pi] interval
                car_angle = (car_angle + (2 * np.pi)) % (2 * np.pi)

                # Retrieve car position
                car_pos = np.array(car.hull.position).reshape((1, 2))
                car_pos_as_point = Point((float(car_pos[:, 0]), float(car_pos[:, 1])))

                # Compute closest point on track to car position (l2 norm)
                distance_to_tiles = np.linalg.norm(
                    car_pos - np.array(self.track)[:, 2:], ord=2, axis=1
                )
                track_index = np.argmin(distance_to_tiles)

                # Check if car is driving on grass by checking inside polygons
                on_grass = not np.array(
                    [
                        car_pos_as_point.within(polygon)
                        for polygon in self.road_poly_shapely
                    ]
                ).any()
                self.driving_on_grass[car_id] = on_grass

                # Find track angle of closest point
                desired_angle = self.track[track_index][1]

                # If track direction reversed, reverse desired angle
                if self.episode_direction == "CW":  # CW direction indicates reversed
                    desired_angle += np.pi

                # Map angle to [0, 2pi] interval
                desired_angle = (desired_angle + (2 * np.pi)) % (2 * np.pi)

                # Compute smallest angle difference between desired and car
                angle_diff = abs(desired_angle - car_angle)
                if angle_diff > np.pi:
                    angle_diff = abs(angle_diff - 2 * np.pi)

                # If car is driving backward and not on grass, penalize car. The
                # backwards flag is set even if it is driving on grass.
                if angle_diff > BACKWARD_THRESHOLD:
                    self.driving_backward[car_id] = True
                    step_reward[car_id] -= K_BACKWARD * angle_diff
                else:
                    self.driving_backward[car_id] = False

            self.prev_reward = self.reward.copy()
            if len(self.track) in self.tile_visited_count:
                done = True

            # The car that leaves the field experiences a reward of -100
            # and the episode is terminated subsequently.
            for car_id, car in enumerate(self.cars):
                x, y = car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done = True
                    step_reward[car_id] = -100

        for car_id, car in enumerate(self.cars):
            if self.driving_on_grass[car_id]:
                self.reset_car(car_id=car_id)
                step_reward[car_id] = 0.0

        step_reward = step_reward.clip(min=0.0)
        return self.state, step_reward, done, {}

    def render(self, mode="human", close=False):
        assert mode in ["human", "state_pixels", "rgb_array"]

        self.renderer.update()

        result = []
        for cur_car_id in range(self.num_agents):
            result.append(self.renderer.render_window(cur_car_id, mode))

        return np.stack(result, axis=0)
