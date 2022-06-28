import random
from gym_multi_deepracer import MultiDeepRacer

NUM_CARS = 1


def my_policy(obs):
    actions = [
        [-1.0, 0.0, 0.0],  # Turn_left
        [+1.0, 0.0, 0.0],  # Turn_right
        #[0.0, 0.0, 0.8],  # Brake
        [0.0, 1.0, 0.0],  # Accelerate
        #[0.0, 0.0, 0.0],  # Do-Nothing
    ]
    action = [random.choice(actions) for _ in range(NUM_CARS)]
    return action


def test_refactoring():
    env = MultiDeepRacer(
        num_agents=NUM_CARS,
        direction="CCW",
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        use_ego_color=False,
    )
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:
        # The actions have to be of the format (num_agents,3)
        # The action format for each car is as in the CarRacing-v0 environment.
        action = my_policy(obs)

        # Similarly, the structure of this is the same as in CarRacing-v0 with an
        # additional dimension for the different agents, i.e.
        # obs is of shape (num_agents, 96, 96, 3)
        # reward is of shape (num_agents,)
        # done is a bool and info is not used (an empty dict).
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

        print("individual scores:", total_reward)
