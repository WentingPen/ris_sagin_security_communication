"""Use exactly the MAPPO actors and action parameterization for a fair baseline."""

from algorithm.ppo.actor import HAPSRISActor, SatelliteActor

__all__ = ["SatelliteActor", "HAPSRISActor"]
