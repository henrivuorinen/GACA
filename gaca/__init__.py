"""GACA: the Gravitational Accretion Clustering Algorithm.

    from gaca import GACA
    model = GACA(gamma_clustering=1.0, random_state=0).fit(X)
    labels = model.assign(X)
"""
from .genesis import GACANode, assign, merge_connected_components, solar_genesis
from .model import GACA

__all__ = ["GACA", "GACANode", "assign", "merge_connected_components", "solar_genesis"]
__version__ = "1.0.0"
