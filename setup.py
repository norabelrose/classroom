from setuptools import setup


setup(
    name="classroom",
    version="0.1",
    description="Preference-based reinforcement learning in PyTorch and JAX with a browser-based GUI.",
    install_requires=[
        "sanic",
        "networkx",
        "numpy",
        "scipy>=1.8.0"  # For sparse linear algebra
    ],
    python_requires=">=3.10",
    extras_require={
        'docs': ['sphinx'],
        'gym': ["gym~=0.22.0"], # 0.22.0 switched to PyGame for rendering
        'jax': ["brax", "jax"],
        'test': ['hypothesis', 'hypothesis-networkx', 'pytest'],
        'torch': ['torch']
    }
)