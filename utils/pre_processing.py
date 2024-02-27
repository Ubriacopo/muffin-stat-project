from __future__ import annotations

import keras
from typing_extensions import deprecated


class AugmentationProcedure:
    def __init__(self):
        self.augmentation_procedure = keras.Sequential([])

    def add_rescaling(self, rescaling_factor: float = 1. / 255) -> AugmentationProcedure:
        self.augmentation_procedure.add(keras.layers.Rescaling(rescaling_factor))
        return self

    def add_random_flip(self, vertical: bool = True, horizontal: bool = True) -> AugmentationProcedure:
        if (vertical or vertical) is False:
            raise Exception("At least one value has to be true.")

        flip_method = "horizontal"
        if vertical and horizontal:  # What we mostly will use.
            flip_method = "horizontal_and_vertical"

        elif vertical:  # If not both of course it has to be vertical
            flip_method = "vertical"

        self.augmentation_procedure.add(keras.layers.RandomFlip(flip_method))
        return self

    def add_random_brightness(self, factor: tuple[float, float] = (-0.5, 0.5),
                              v_range: tuple[float, float] = (0, 1)) -> AugmentationProcedure:
        self.augmentation_procedure.add(keras.layers.RandomBrightness(factor=factor, value_range=v_range))
        return self

    def add_random_rotation(self, random_rotation: float = 0.2) -> AugmentationProcedure:
        self.augmentation_procedure.add(keras.layers.RandomRotation(random_rotation))
        return self

    def add_normalization(self, mean: float = 0., variance: float = 1.) -> AugmentationProcedure:
        self.augmentation_procedure.add(keras.layers.Normalization(mean=mean, variance=variance))
        return self


class AugmentationFactory:
    @staticmethod
    @deprecated
    def make_complete_procedure(rescaling_factor: float = 1. / 255) -> AugmentationProcedure:
        """
        :deprecated: use make_augmentation_procedure() instead
        :param rescaling_factor:
        :return:
        """
        return AugmentationProcedure().add_rescaling(rescaling_factor).add_random_flip()

    @staticmethod
    def make_complete_procedure_keras() -> AugmentationProcedure:
        return AugmentationProcedure().add_random_flip().add_random_brightness()
