import os
import torch
from transformers.trainer import WEIGHTS_NAME

from llmtuner.extras.logging import get_logger


logger = get_logger(__name__)


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    vhead_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(vhead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    vhead_params = torch.load(vhead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
    model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
    model.register_buffer("default_head_weight", torch.zeros_like(vhead_params["v_head.summary.weight"]), persistent=False)
    model.register_buffer("default_head_bias", torch.zeros_like(vhead_params["v_head.summary.bias"]), persistent=False)
    return True

##################################
# MODIFIED for URM Dropout
##################################
def load_valuehead_params2(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:

    load1 = load_valuehead_params(model, checkpoint_dir)

    vhead_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(vhead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    vhead_params = torch.load(vhead_file, map_location="cpu")
    if "interm_seq.lin3.weight" in vhead_params:
        model.register_buffer("interm_seq_weight", vhead_params["interm_seq.lin3.weight"], persistent=False)
        model.register_buffer("default_interm_seq_weight", torch.zeros_like(vhead_params["interm_seq.lin3.weight"]), persistent=False)
    return True and load1