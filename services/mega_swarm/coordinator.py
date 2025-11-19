"""
Mega Swarm Coordinator - Bridge to Enhanced Mega Coordinator
"""

from ..swarm.coordination.enhanced_mega_coordinator import (
    EnhancedMegaSwarmCoordinator as MegaSwarmCoordinator
)
from ..swarm.unified_swarm_system import SwarmScale, SwarmObjective, UnifiedGoal as Goal

__all__ = [
    'MegaSwarmCoordinator',
    'SwarmScale',
    'SwarmObjective',
    'Goal'
]



