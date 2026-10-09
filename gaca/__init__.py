"""GACA: the Gravitational Accretion Clustering Algorithm.

    from gaca import AutoGACA            # any table: preprocessing and gamma chosen
    auto = AutoGACA().fit(df)
    auto.result_, auto.report("report.html")

    from gaca import GACA                # the algorithm itself
    model = GACA(gamma_clustering=1.0, random_state=0).fit(X)
    labels = model.assign(X)
"""
from .genesis import (GACANode, assign, kernel_pull, merge_connected_components,
                      saddle_link, solar_genesis)
from .model import GACA
from .auto import AutoGACA, Preprocessor, select_gamma

__all__ = ["GACA", "AutoGACA", "Preprocessor", "select_gamma", "GACANode", "assign",
           "kernel_pull", "merge_connected_components", "saddle_link", "solar_genesis"]
__version__ = "1.1.0"
