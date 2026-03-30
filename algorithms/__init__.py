"""
algorithms/ — Decentralized Personalized Learning algorithms.

Usage:
    from algorithms import build_algorithm
    from dl.config import DL_CFG
    algo = build_algorithm(DL_CFG)
    algo.setup(vehicles)
"""

from algorithms.base import DLAlgorithm, LINK_SIDELINK, LINK_INTERNET


def build_algorithm(cfg: dict) -> DLAlgorithm:
    """Instantiate and return the algorithm specified by cfg["ALGORITHM"]."""
    name = cfg["ALGORITHM"]

    if name == "FedAvg":
        from algorithms.fedavg.algorithm import FedAvgAlgorithm
        return FedAvgAlgorithm()

    if name == "D-PSGD":
        from algorithms.dsgd.algorithm import DSGDAlgorithm
        return DSGDAlgorithm()

    if name == "DPFL":
        from algorithms.dpfl.algorithm import DPFLAlgorithm
        return DPFLAlgorithm(update_every=int(cfg.get("DPFL_UPDATE_EVERY", 10)))

    raise ValueError(
        f"Unknown algorithm {name!r}. "
        f"Choose from: FedAvg, D-PSGD, DPFL"
    )


__all__ = ["DLAlgorithm", "LINK_SIDELINK", "LINK_INTERNET", "build_algorithm"]
