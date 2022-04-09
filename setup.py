from setuptools import setup


setup(
    name="classroom",
    version="0.1",
    description="Preference-based reinforcement learning in PyTorch and JAX with a browser-based GUI.",
    install_requires=[
        "sanic",
        "networkx",
        "numpy"
    ],
    python_requires=">=3.10",
    extras_require={
        'gym': ["gym~=0.22.0"], # 0.22.0 switched to PyGame for rendering
        'jax': ["brax", "jax"],
        'test': ['hypothesis', 'hypothesis-networkx', 'pytest'],
        'torch': ['torch']
    }
)