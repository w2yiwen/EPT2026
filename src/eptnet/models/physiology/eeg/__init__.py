"""EEG feature encoders."""

from .eegnet import EEGNetFeatureEncoder, EEGNetPreprocessor
from .eegnet_v4.vendor.eegnet_core import EEGNetV4

__all__ = ["EEGNetFeatureEncoder", "EEGNetPreprocessor", "EEGNetV4"]
