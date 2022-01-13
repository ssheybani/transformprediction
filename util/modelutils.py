import torch
from torch import nn
from tqdm.notebook import tqdm as tqdm
from matplotlib import pyplot as plt

def show_progress_bars(enable=True):
    """
    show_progress_bars()

    Enabled or disables tqdm progress bars.

    Optional args:
    - enabled (bool or str): progress bar setting ("reset" to previous)
    """

    if enable == "reset":
        if hasattr(tqdm, "_patch_prev_enable"):
            enable = tqdm._patch_prev_enable
        else:
            enable = True

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not(enable))
    tqdm._patch_prev_enable = not(enable)


def get_model_device(model):
    """
    get_model_device(model)

    Returns the device that the first parameters in a model are stored on.

    N.B.: Different components of a model can be stored on different devices. 
    Thisfunction does NOT check for this case, so it should only be used when 
    all model components are expected to be on the same device.

    Required args:
    - model (nn.Module): a torch model

    Returns:
    - first_param_device (str): device on which the first parameters of the 
        model are stored
    """
    

    if len(list(model.parameters())):
        first_param_device = next(model.parameters()).device
    else:
        first_param_device = "cpu" # default if the model has no parameters

    return first_param_device
