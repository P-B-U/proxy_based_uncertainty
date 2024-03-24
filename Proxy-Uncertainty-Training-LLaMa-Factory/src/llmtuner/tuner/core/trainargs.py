from transformers import Seq2SeqTrainingArguments
from dataclasses import field
from dataclasses import dataclass, field

@dataclass# @add_start_docstrings(Seq2SeqTrainingArguments.__doc__)
class Seq2SeqTrainingArgumentsURM(Seq2SeqTrainingArguments):
    """
    Args:
    """
    sequential_sampler: bool = field(default=True, metadata={"help": "Whether to use Sequential Sampler or not."})
    use_balent_loss1: bool = field(default=False, metadata={"help": "Whether to use Balanced Entropy Loss1 or not."})