"""Built-in dataset adapters for the training SDK.

Each adapter implements :class:`workflow.training.DatasetAdapter` for one
public credit dataset:

  * :class:`UCIAdapter`     — UCI Default of Credit Card Clients (Taiwan, 30K)
  * :class:`HMDAAdapter`    — HMDA Mortgage Data (US, sampled to 100K)
  * :class:`BondoraAdapter` — Bondora P2P Lending (Estonia / EU, ~100K)
"""
from .bondora import BondoraAdapter
from .hmda import HMDAAdapter
from .uci import UCIAdapter

__all__ = ["UCIAdapter", "HMDAAdapter", "BondoraAdapter"]
