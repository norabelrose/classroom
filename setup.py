from setuptools import setup


setup(
    name="whisper",
    version="0.1",
    description="Preference-based reinforcement learning in PyTorch and JAX with a browser-based GUI.",
    requires=[
        "gym",
        "flask",
    ]
)