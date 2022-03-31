from setuptools import setup


setup(
    name="classroom",
    version="0.1",
    description="Preference-based reinforcement learning in PyTorch and JAX with a browser-based GUI.",
    requires=[
        "cherrypy",
        "gym~=0.22.0",  # 0.22.0 switched to PyGame for rendering
        "networkx"
    ]
)