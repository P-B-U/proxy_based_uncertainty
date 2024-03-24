import torch
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import (
    has_length,
)

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class CustomDPOTrainer(DPOTrainer):

    def __init__(
        self,
        beta: float,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]] = None,
        disable_dropout: Optional[bool] = True,
        **kwargs
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.ref_model = ref_model
        self.use_dpo_data_collator = True # hack to avoid warning
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.beta = beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dpo_counter = 0

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model, = self.accelerator._prepare_deepspeed(self.ref_model)
                self.ref_model.eval()
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        ####################################################################################
        ## ADDED FOR USE BALENT TERM IN DPO LOSS
        ####################################################################################
        train_arguments = kwargs.get('args', None)
        if train_arguments:
            self.sequential_sampler = train_arguments.sequential_sampler
            self.use_balent_loss1 = train_arguments.use_balent_loss1
            self.balanced_entropy = []
            if self.use_balent_loss1:
                self.balanced_entropy = [self.train_dataset[i]['uncertainties']['balanced_entropy'] for i in range(len(self.train_dataset))]
            self.balanced_entropy = np.array(self.balanced_entropy)
        else:
            self.sequential_sampler = True
            self.use_balent_loss1 = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.sequential_sampler:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None
            else:
                return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler()

    def concatenated_forward(
        self,
        model: Optional[torch.nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()}) # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            return_dict=True
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            False,
            batch.get('uncertainties', None)  
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics
    #####################################
    ## MODIFIED FOR UDPO BALENT LOSS
    #####################################
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        uncertainty = 1.0
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios
        if self.use_balent_loss1:
            uncertainty = uncertainty[:logits.shape[0]]
            loss_factor =  2.71828182846-torch.exp(uncertainty)
            loss_factor.detach()
            losses = -loss_factor*F.logsigmoid(self.beta * logits)
        else:
            losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards