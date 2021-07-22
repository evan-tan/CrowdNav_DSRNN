from setuptools import setup


setup(
    name="crowdnav",
    version="0.0.1",
    packages=[
        "crowd_nav",
        "crowd_nav.configs",
        "crowd_nav.policy",
        "crowd_sim",
        "crowd_sim.envs",
        "crowd_sim.envs.utils",
    ],
    install_requires=[
        "gym",
        "tensorflow",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "torch",
        "torchvision",
        "scipy",
    ],
)
