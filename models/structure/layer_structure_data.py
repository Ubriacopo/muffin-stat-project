from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HiddenLayerStructure:
    units: int
    following_dropout: float | None


@dataclass
class ConvLayerStructure:
    kernel_size: tuple[int, int]
    filters: int


@dataclass
class PoolLayerStructure:
    pool_size: tuple[int, int]
    stride: int
