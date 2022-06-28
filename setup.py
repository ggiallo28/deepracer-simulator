from setuptools import setup

setup(
    name='gym_multi_deepracer',
    version='0.0.1',
    url='https://github.com/ggiallo28/deepracer-simulator',
    description='Gym Multi DeepRacer Environment',
    packages=['gym_multi_deepracer'],
    install_requires=[
        'box2d-py~=2.3.8',
        'shapely~=1.8.2',
        'numpy>=1.23.0',
        'gym~=0.7.4',
        'pyglet==1.5.26',
    ]
)