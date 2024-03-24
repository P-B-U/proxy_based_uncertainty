import os
import json
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from transformers import Trainer

from llmtuner.extras.logging import get_logger

import numpy as np

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from transformers.modeling_utils import PreTrainedModel


logger = get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""
    Inherits PeftTrainer to compute pairwise loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True # override property to return eval_loss

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple. 
        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        # Compute rewards
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        if values.size(0) != inputs["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)

        # Split the inputs and rewards into two parts, chosen and rejected
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
        chosen_attn_mask, rejected_attn_mask = (
            inputs["attention_mask"][:batch_size], inputs["attention_mask"][batch_size:]
        )
        chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
        chosen_scores, rejected_scores = [], []

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
        loss = 0
        for i in range(batch_size):
            chosen_length = chosen_attn_mask[i].nonzero()[-1] + 1
            rejected_length = rejected_attn_mask[i].nonzero()[-1] + 1
            check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            assert div_index > 0
            chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
            rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
            if return_outputs: # use the score on the EOS token for inference
                chosen_scores.append(chosen_rewards[i, chosen_length-1])
                rejected_scores.append(rejected_rewards[i, rejected_length-1])

            loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

        loss = loss / batch_size
        if return_outputs:
            chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)
            return loss, [loss, chosen_scores, rejected_scores]

        return loss

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))
            writer.write("\n".join(res))


################################
#### ADDED FOR URM DROPOUT
################################
class PairwiseTrainerURM(PairwiseTrainer):
    r"""
    Inherits PairwiseTrainer to compute pairwise loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number_of_dropout_samples = 25
        self.save_tensor = False
        print("***ARG GROUP***")
        # print(args)
        train_arguments = kwargs.get('args', None)
        print(train_arguments)
        if has_length(self.train_dataset):
            print(self.train_dataset[0])
        if train_arguments:
            self.sequential_sampler = train_arguments.sequential_sampler
            self.use_balent_loss1 = False
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

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple. 
        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        # Compute rewards
        ############################
        ## MODIFIED FOR DROPOUT
        ############################
        if return_outputs:
            model.v_head.dropout.train()
            model.interm_seq.drp2.train()

            ## just in case
            uncertainties = inputs.pop('uncertainties', None)
            sample = []
            with torch.no_grad():
                for idx in range(self.number_of_dropout_samples):
                    _, _, values = model(**inputs, output_hidden_states=True, return_dict=True) #BxL
                    sample.append(values)

            collected_outputs = torch.stack(sample)
            collected_outputs = collected_outputs.permute(1,0,2) # B X M X L
            values = torch.mean(collected_outputs, dim=1)
        else:
            uncertainty = inputs.pop('uncertainties', None)
            _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)

        if values.size(0) != inputs["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)

        # Split the inputs and rewards into two parts, chosen and rejected
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_input_ids, rejected_input_ids = inputs["input_ids"][:batch_size], inputs["input_ids"][batch_size:]
        chosen_attn_mask, rejected_attn_mask = (
            inputs["attention_mask"][:batch_size], inputs["attention_mask"][batch_size:]
        )
        chosen_rewards, rejected_rewards = values[:batch_size], values[batch_size:]
        chosen_scores, rejected_scores = [], []
        ## ADDED
        chosen_outputs, rejected_outputs = [], []

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # Inspired by: https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
        loss = 0
        for i in range(batch_size):
            chosen_length = chosen_attn_mask[i].nonzero()[-1] + 1
            rejected_length = rejected_attn_mask[i].nonzero()[-1] + 1
            check_divergence = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero()

            if len(check_divergence) == 0:
                end_index = chosen_length
                div_index = end_index - 1
            else:
                end_index = max(chosen_length, rejected_length)
                div_index = check_divergence[0]

            assert div_index > 0
            chosen_trunc_rewards = chosen_rewards[i, div_index:end_index]
            rejected_trunc_rewards = rejected_rewards[i, div_index:end_index]
            if return_outputs: # use the score on the EOS token for inference
                chosen_scores.append(chosen_rewards[i, chosen_length-1])
                rejected_scores.append(rejected_rewards[i, rejected_length-1])
                ## ADDED
                chosen_outputs.append(collected_outputs[i, :, chosen_length-1:chosen_length])
                rejected_outputs.append(collected_outputs[batch_size+i, :, rejected_length-1:rejected_length])

            loss += -torch.nn.functional.logsigmoid(chosen_trunc_rewards - rejected_trunc_rewards).mean()

        loss = loss / batch_size
        if return_outputs:
            chosen_scores, rejected_scores = torch.stack(chosen_scores), torch.stack(rejected_scores)

            ##############################################################################
            ## Uncertainty Calculations
            ##############################################################################
            chosen_outputs = torch.stack(chosen_outputs).detach().cpu()
            rejected_outputs = torch.stack(rejected_outputs).detach().cpu()
            ##############################################################################
            ## for difference
            diff = chosen_outputs - rejected_outputs
            diff = torch.mean(diff, dim=2)

            logits = torch.nn.functional.logsigmoid(diff.to(torch.float32))
            logits = torch.log(torch.stack([torch.exp(logits), 1.-torch.exp(logits)+1e-128], dim=2))
            logits = torch.nn.functional.log_softmax(logits, dim=2)

            m = torch.mean(diff, dim=1)
            m = m.reshape(m.shape[0], 1).to(torch.float32)
            s = torch.std(diff, dim=1)
            s = s.reshape(s.shape[0], 1).to(torch.float32)
            ##############################################################################
            ## for chosen
            chosen_outputs = torch.mean(chosen_outputs, dim=2)
            m_chosen_outputs = torch.mean(chosen_outputs, dim=1)
            m_chosen_outputs = m_chosen_outputs.reshape(m.shape[0], 1).to(torch.float32)
            s_chosen_outputs = torch.std(chosen_outputs, dim=1)
            s_chosen_outputs = s.reshape(s_chosen_outputs.shape[0], 1).to(torch.float32)
            del chosen_outputs

            ##############################################################################
            ## for rejection
            rejected_outputs = torch.mean(rejected_outputs, dim=2)
            m_rejected_outputs = torch.mean(rejected_outputs, dim=1)
            m_rejected_outputs = m_rejected_outputs.reshape(m.shape[0], 1).to(torch.float32)
            s_rejected_outputs = torch.std(rejected_outputs, dim=1)
            s_rejected_outputs = s.reshape(s_rejected_outputs.shape[0], 1).to(torch.float32)
            del rejected_outputs
            ##############################################################################
            uncertainties = {}
            if self.save_tensor:
                uncertainties['Diff'] = diff.float()
            del diff
            print(m, s)
            uncertainties['BalEnt'] = balanced_entropy(m, s)
            uncertainties['Epistemic'] = mutual_information(logits)
            uncertainties['Aleatoric'] = aleatoric_uncertainty_acquisition_function(logits)
            uncertainties['Beta_BalEnt'] = beta_balanced_entropy(logits)
            uncertainties['Chosen_BalEnt'] = balanced_entropy(m_chosen_outputs, s_chosen_outputs)
            uncertainties['Rejected_BalEnt'] = balanced_entropy(m_rejected_outputs, s_rejected_outputs)
            return loss, [loss, chosen_scores, rejected_scores, uncertainties]

        return loss

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        ############################################################################################################################
        ## MODIFIED
        chosen_scores, rejected_scores, uncertainties = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score, bal_ent, epis, alea, beta_bal_ent, chosen_balent, rejected_balent in zip(chosen_scores, rejected_scores, uncertainties['BalEnt'], 
                                                 uncertainties['Epistemic'], uncertainties['Aleatoric'], uncertainties['Beta_BalEnt'], 
                                                 uncertainties['Chosen_BalEnt'], uncertainties['Rejected_BalEnt'] ):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2),
                                       "balanced_entropy": round(float(bal_ent), 5),
                                       "beta_balanced_entropy": round(float(beta_bal_ent), 5),
                                       "epistemic_uncertainty": round(float(epis), 5),
                                       "aleatoric_uncertainty": round(float(alea), 5),
                                       "chosen_balanced_entropy": round(float(chosen_balent), 5),
                                       "rejected_balanced_entropy": round(float(rejected_balent), 5),}))
        ############################################################################################################################
            writer.write("\n".join(res))

        if self.save_tensor:
            output_tensor_file = os.path.join(self.args.output_dir, "generated_diff.npy")
            diff_array = []
            for diff in uncertainties['Diff']:
                diff_array.append(diff)
            np.save(output_tensor_file, np.vstack(diff_array))


import math

# density of sigmoid transformed gaussian distribution
def f(x, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    return (1/s) * (1/(x*(1-x))) * torch.exp(-((torch.log(x/(1-x))-m)/s)**2 / 2) / (math.sqrt(2 * math.pi))

# posterior density of f
def fplus(x, ef_val, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    return x*f(x, m, s, a, b, n) / ef_val.reshape(ef_val.shape[0], 1)

# numerical integration, use Trapezoidal method
def tintf(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    y = f(x, m, s)
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n])
    result[result>b] = b
    result[result<a] = a
    return result

# expectation of f
def ef(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = f(x, m, s)
    idx = fv <= 1e-10

    rem = 1 - tintf(m, s, a, b, n)
    leftover = rem * n
    idx = rem <= 0
    leftover[idx] = 0

    y = x* fv
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n]) + leftover * h
    result[result>b] = b
    result[result<a] = a
    return result

# differential entropy of posterior f
def hfplus(ef_val, m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = fplus(x, ef_val, m, s)
    idx = fv <= 1e-10
    logfv = torch.log(fv)
    logfv[idx] = 0

    rem = 1 - tintf(m, s, a, b, n)
    leftover = -rem * n * torch.log(rem * n)
    idx = rem <= 0
    leftover[idx] = 0

    y = -fv * logfv
    result = (h / 2) * (2 * torch.sum(y, dim = 1) - y[:, 0] - y[:, n]) + leftover * h
    return result

# Shannon entropy of Ef
def hef(m=0, s=1, a=0.00001, b=0.99999, n=10000):
    h = (b - a) / n
    x = torch.linspace(a, b, n + 1).to(m.device)
    x = torch.stack([x for i in range(m.shape[0])])
    fv = f(x, m, s)
    
    y = fv * x
    p = (h / 2) * (2 * torch.sum(y, dim=1) - y[:, 0] - y[:, n])

    p[p>b] = b
    p[p<a] = a
    
    result = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    return result

def balanced_entropy(m, s):
    ef_val = ef(m,s)
    bal_ent = (ef_val*hfplus(ef_val, m,s) + (1-ef_val)*hfplus(1-ef_val, -m,s) + hef(m,s))/(hef(m,s)+0.6931472)
    return bal_ent

def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits), dim=dim, keepdim=keepdim)

def logit_mean(logits, dim: int, keepdim: bool = False):
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])

def mutual_information(logits_B_K_C):
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B

    idx = mutual_info_B < 1e-9
    mutual_info_B[idx] = mutual_info_B[idx] * 0 + 1e-9

    return mutual_info_B

def aleatoric_uncertainty_acquisition_function(logits_B_K_C):
    pred_entropy = entropy(logit_mean(logits_B_K_C, dim=1, keepdim=False), dim=-1)
    bald = mutual_information(logits_B_K_C)

    return pred_entropy - bald

def beta_balanced_entropy(logits_B_K_C):

    mjent = marginalized_posterior_entropy(logits_B_K_C)
    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    balent = (mjent) / (mean_entropy_B+0.69314718056)

    return balent  


def marginalized_posterior_entropy(logits_B_K_C):
    
    unlogits_mean_B_C, unlogits_var_B_C = unlogit_meanvar(logits_B_K_C, dim=1)
    idx = unlogits_mean_B_C < 1e-9
    unlogits_mean_B_C[idx] = 1e-9
    idx = unlogits_var_B_C < 1e-9
    unlogits_var_B_C[idx] = 1e-9#1e-9

    all_alpha = unlogits_mean_B_C*unlogits_mean_B_C*(1-unlogits_mean_B_C)/unlogits_var_B_C-unlogits_mean_B_C
    all_beta = (1/unlogits_mean_B_C -1)*all_alpha

    hpp = torch.log(beta_function(all_alpha+1, all_beta)) - all_alpha*torch.digamma(all_alpha+1) - (all_beta-1)*torch.digamma(all_beta) + (all_alpha+all_beta-1)*torch.digamma(all_alpha+all_beta+1)
    all_beta_entropy = unlogits_mean_B_C*hpp
    idx = all_beta_entropy>0
    print('betapp flag')
    print(all_alpha[idx], all_beta[idx])
    print(torch.sum(idx))
    all_beta_entropy[idx] = 0
    baba_information_B = torch.sum(all_beta_entropy, dim=1)
    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)
    baba_information_B += mean_entropy_B

    ###############################################################################
    ## treat high confidence case
    baba_information_B[torch.where(torch.isinf(baba_information_B))]=-99999999999
    baba_information_B[torch.where(torch.isnan(baba_information_B))]=-99999999999

    return baba_information_B

def beta_function(x, y):
    return torch.exp(torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y)) + 1e-256

def unlogit_meanvar(logits, dim: int, keepdim: bool = False):
    r"""Computes mean & variance

    We pass in logits.
    """
    unlogit_ave = torch.exp(logit_mean(logits, dim, keepdim))
    unlogit_var = torch.var(torch.exp(logits), dim=dim, keepdim=keepdim)

    return unlogit_ave, unlogit_var
############################################################################################################################