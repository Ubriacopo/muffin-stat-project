from __future__ import annotationsKerasTuner

from dataclasses import dataclass


@dataclass
class HiddenLayerStructure:
    units: int


@dataclass
class DropoutLayerStructure:
    rate: float


@dataclass
class ConvLayerStructure:
    kernel_size: tuple[int, int]
    filters: int


@dataclass
class PoolLayerStructure:
    pool_size: tuple[int, int]
    stride: int

    @staticmethod
    def default():
        return PoolLayerStructure(pool_size=(2, 2), stride=2)
