"""Stub for checkpoint unpickling. The real class needs Lightning;
this minimal version lets torch.load resolve pickled references."""

import torch.nn as nn


def get_w2v2ph_schedule(*args, **kwargs):
    pass


class PhoneRecognitionModel(nn.Module):
    pass
