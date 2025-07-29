from setuptools import setup, find_packages

setup(
    name="humanoid_climb",
    version="0.1.0",
    description="Humanoid Climbing Reinforcement Learning Environment",
    author="Humanoid Climb Team",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.21.0",
        "pybullet>=3.2.5",
        "stable-baselines3>=2.0.0",
        "torch>=1.13.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
