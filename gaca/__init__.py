"""GACA: the Gravitational Accretion Clustering Algorithm.

    from gaca import GACA
    model = GACA(gamma_clustering=1.0, random_state=0).fit(X)
    labels = model.assign(X)
"""
from .genesis import (GACANode, assign, kernel_pull, merge_connected_components,
                      saddle_link, solar_genesis)
from .model import GACA

__all__ = ["GACA", "GACANode", "assign", "kernel_pull", "merge_connected_components",
           "saddle_link", "solar_genesis"]
__version__ = "1.1.0"
