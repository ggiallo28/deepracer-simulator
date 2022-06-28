from pyglet.window import key
from multi_deepracer import MultiDeepRacer
import numpy as np

NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed

# Specify key controls for cars
CAR_CONTROL_KEYS = [
    [key.LEFT, key.RIGHT, key.UP, key.DOWN],
    [key.A, key.D, key.W, key.S],
]

a = np.zeros((NUM_CARS, 3))


def key_press(k, mod):
    global restart, stopped, CAR_CONTROL_KEYS
    if k == 0xFF1B:
        stopped = True  # Terminate on esc.
    if k == 0xFF0D:
        restart = True  # Restart on Enter.

    # Iterate through cars and assign them control keys (mod num controllers)
    for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]:
            a[i][0] = -1.0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]:
            a[i][0] = +1.0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:
            a[i][1] = +1.0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:
            a[i][2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    global CAR_CONTROL_KEYS

    # Iterate through cars and assign them control keys (mod num controllers)
    for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0] and a[i][0] == -1.0:
            a[i][0] = 0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1] and a[i][0] == +1.0:
            a[i][0] = 0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:
            a[i][1] = 0
        if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:
            a[i][2] = 0


env = MultiDeepRacer(NUM_CARS)
env.render()
for viewer in env.viewer:
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release
record_video = False
if record_video:
    from gym.wrappers.monitor import Monitor

    env = Monitor(env, "/tmp/video-test", force=True)
isopen = True
stopped = False
while isopen and not stopped:
    env.reset()
    total_reward = np.zeros(NUM_CARS)
    steps = 0
    restart = False
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            print(
                "\nActions: "
                + str.join(" ", [f"Car {x}: " + str(a[x]) for x in range(NUM_CARS)])
            )
            print(f"Step {steps} Total_reward " + str(total_reward))
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.savefig("test.jpeg")
        steps += 1
        isopen = env.render().all()
        # if stopped or done or restart or isopen == False:
        #    break
env.close()
