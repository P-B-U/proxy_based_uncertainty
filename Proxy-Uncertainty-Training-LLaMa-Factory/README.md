### Our repo is based on [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory).
#### We can adopt the same approach as outlined by the LLaMa-Factory.

Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken
- jieba, rouge-chinese and nltk (used at evaluation)
- gradio and matplotlib (used in web_demo.py)
- uvicorn, fastapi and sse-starlette (used in api_demo.py)

And **powerful GPUs**!

## Getting Started

### Data Preparation (optional)

Please refer to `data/example_dataset` for checking the details about the format of dataset files. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset. About the format of this file, please refer to `data/README.md`.

### Dependence Installation (optional)

```bash
git clone https://github.com/hiyouga/LLaMA-Efficient-Tuning.git
conda create -n llama_etuning python=3.10
conda activate llama_etuning
cd LLaMA-Efficient-Tuning
pip install -r requirements.txt
```

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you will be required to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.1.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

### All-in-one Web UI

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_web.py
```

We strongly recommend using the all-in-one Web UI for newcomers since it can also generate training scripts **automatically**.

> [!WARNING]
> Currently the web UI only supports training on **a single GPU**.

### Train on a single GPU

> [!IMPORTANT]
> If you want to train models on multiple GPUs, please refer to [Distributed Training](#distributed-training).

#### Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset wiki_demo \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_pt_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

#### Reward Modeling

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
    --output_dir path_to_rm_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

#### PPO Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
    --reward_model path_to_rm_checkpoint \
    --output_dir path_to_ppo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

#### DPO Training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir path_to_sft_checkpoint \
    --output_dir path_to_dpo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

### Distributed Training

#### Use Huggingface Accelerate

```bash
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

<details><summary>Example config for LoRA training</summary>

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</details>

#### Use DeepSpeed

```bash
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    ... # arguments (same as above)
```

<details><summary>Example config for full-parameter training with DeepSpeed ZeRO-2</summary>

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },  
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
```

</details>

### Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

### API Demo

```bash
python src/api_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

> [!NOTE]
> Visit `http://localhost:8000/docs` for API documentation.

### CLI Demo

```bash
python src/cli_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### Web Demo

```bash
python src/web_demo.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint
```

### Evaluation (BLEU and ROUGE_CHINESE)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

> [!NOTE]
> We recommend using `--per_device_eval_batch_size=1` and `--max_target_length 128` at 4/8-bit evaluation.

### Predict

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights:

- [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
- [LLaMA-2](https://ai.meta.com/llama/license/)
- [BLOOM](https://huggingface.co/spaces/bigscience/license)
- [Falcon](LICENSE)
- [Baichuan](https://huggingface.co/baichuan-inc/baichuan-7B/resolve/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)
- [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/Baichuan%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)
- [InternLM](https://github.com/InternLM/InternLM#open-source-license)
- [Qwen](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/LICENSE)
- [XVERSE](https://github.com/xverse-ai/XVERSE-13B/blob/main/MODEL_LICENSE.pdf)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B/blob/main/MODEL_LICENSE)

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{llama-efficient-tuning,
  title = {LLaMA Efficient Tuning},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Efficient-Tuning}},
  year = {2023}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [QLoRA](https://github.com/artidoro/qlora), [FastChat](https://github.com/lm-sys/FastChat) and [OpenChatKit](https://github.com/togethercomputer/OpenChatKit). Thanks for their wonderful works.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Efficient-Tuning&type=Date)
