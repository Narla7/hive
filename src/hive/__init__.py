"""HIVE: an evolutionary harness for day-trading agents.

Fitness is realized net P&L from a double-entry ledger. The model proposes
actions; the gates and the broker decide what actually happens.
"""

__version__ = "0.1.0"

from .broker import BrokerConfig, PaperBroker
from .evolution import Config, run
from .fitness import Episode, Fitness, evaluate
from .gates import GateConfig, Gates
from .genome import Genome, Policy, hand_seeded, mutate, crossover
from .ledger import Ledger
from .market import Bar, SimulatedMarket, FileMarket

__all__ = [
    "BrokerConfig", "PaperBroker", "Config", "run", "Episode", "Fitness",
    "evaluate", "GateConfig", "Gates", "Genome", "Policy", "hand_seeded",
    "mutate", "crossover", "Ledger", "Bar", "SimulatedMarket", "FileMarket",
]
