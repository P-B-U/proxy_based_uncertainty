###################################################
## Full batch is written by Jae Oh
###################################################
import os
import subprocess

settings_dict = {}
settings_dict['hh_train_random'] = {
    'dataset': 'hh_train_random',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None", #"/data1/data/proxy_uncertainty/results/sima/sharegpt_short_sft",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_hh_rm/udpo_cosine/sharegpt2hh_random',
    # 'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_regular_train'] = {
    'dataset': 'dolly_preference_data_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_comparison_train'] = {
    'dataset': 'alpaca_comparison_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/train_alpaca_lr4',
    'learning_rate': '1e-4'
}

##########################################################################################

settings_dict['dolly_train_balent_up'] = {
    'dataset': 'dolly_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_balent_down'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_epistemic_up'] = {
    'dataset': 'dolly_train_epistemic_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_epistemic_down'] = {
    'dataset': 'dolly_train_epistemic_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_down_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_aleatoric_up'] = {
    'dataset': 'dolly_train_aleatoric_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_aleatoric_down'] = {
    'dataset': 'dolly_train_aleatoric_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_down_lr4',
    'learning_rate': '1e-4'
}

#########################################################################################
##########################################################################################
# 2nd epoch
settings_dict['dolly_train_2'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_2_lr43',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_balent_up2'] = {
    'dataset': 'dolly_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_balent_down2'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_epistemic_up2'] = {
    'dataset': 'dolly_train_epistemic_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_epistemic_down2'] = {
    'dataset': 'dolly_train_epistemic_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_down2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_aleatoric_up2'] = {
    'dataset': 'dolly_train_aleatoric_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_train_aleatoric_down2'] = {
    'dataset': 'dolly_train_aleatoric_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_down2_lr4',
    'learning_rate': '1e-4'
}

#########################################################################################

settings_dict['alpaca_train_balent_up'] = {
    'dataset': 'alpaca_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_balent_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_balent_down'] = {
    'dataset': 'alpaca_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_balent_down_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_epistemic_up'] = {
    'dataset': 'alpaca_train_epistemic_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_epistemic_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_epistemic_down'] = {
    'dataset': 'alpaca_train_epistemic_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_epistemic_down_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_aleatoric_up'] = {
    'dataset': 'alpaca_train_aleatoric_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_aleatoric_up_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_aleatoric_down'] = {
    'dataset': 'alpaca_train_aleatoric_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_aleatoric_down_lr4',
    'learning_rate': '1e-4'
}
#########################################################################################
##########################################################################################
# 2nd epoch
settings_dict['alpaca_train_2'] = {
    'dataset': 'alpaca_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_2_lr4',
    'learning_rate': '1e-4'
}


settings_dict['alpaca_train_balent_up2'] = {
    'dataset': 'alpaca_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_balent_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_balent_down2'] = {
    'dataset': 'alpaca_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_balent_down2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_epistemic_up2'] = {
    'dataset': 'alpaca_train_epistemic_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_epistemic_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_epistemic_down2'] = {
    'dataset': 'alpaca_train_epistemic_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_epistemic_down2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_aleatoric_up2'] = {
    'dataset': 'alpaca_train_aleatoric_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_aleatoric_up2_lr4',
    'learning_rate': '1e-4'
}

settings_dict['alpaca_train_aleatoric_down2'] = {
    'dataset': 'alpaca_train_aleatoric_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/alpaca_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/alpaca_train_aleatoric_down2_lr4',
    'learning_rate': '1e-4'
}

#########################################################################################

settings_dict['dolly_regular_balent'] = {
    'dataset': 'dolly_preference_data_balent',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_balent_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_balent'] = {
    'dataset': 'dolly_preference_data_rev_balent',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_balent_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_epistemic'] = {
    'dataset': 'dolly_preference_data_epistemic',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_epistemic_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_epistemic'] = {
    'dataset': 'dolly_preference_data_rev_epistemic',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_epistemic_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_aletoric'] = {
    'dataset': 'dolly_preference_data_aleatoric',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_aleatoric_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_aleatoric'] = {
    'dataset': 'dolly_preference_data_rev_aleatoric',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_aleatoric_lr4',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_2nd'] = {
    'dataset': 'dolly_preference_data',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_balent_2nd'] = {
    'dataset': 'dolly_preference_data_balent',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_balent_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_balent_2nd'] = {
    'dataset': 'dolly_preference_data_rev_balent',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_balent_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_epistemic_2nd'] = {
    'dataset': 'dolly_preference_data_epistemic',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_epistemic_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_epistemic_2nd'] = {
    'dataset': 'dolly_preference_data_rev_epistemic',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_epistemic_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_aleatoric_2nd'] = {
    'dataset': 'dolly_preference_data_aleatoric',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_aleatoric_lr4_2nd',
    'learning_rate': '1e-4'
}

settings_dict['dolly_regular_rev_aleatoric_2nd'] = {
    'dataset': 'dolly_preference_data_rev_aleatoric',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm/output_dolly_preference_data_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm/output_dolly_preference_data_rev_aleatoric_lr4_2nd',
    'learning_rate': '1e-4'
}

# settings = ['dolly_regular_2nd',]# 'dolly_regular_balent_2nd', 'dolly_regular_rev_balent_2nd']
# settings = ['dolly_regular_epistemic', 'dolly_regular_rev_epistemic','dolly_regular_aletoric',]
# settings = [ 'dolly_regular_rev_aleatoric',]
# settings = ['dolly_regular_epistemic_2nd', 'dolly_regular_rev_epistemic_2nd', ]
# settings = ['dolly_regular_aleatoric_2nd','dolly_regular_rev_aleatoric_2nd',]
# settings = ['dolly_regular_train', 'alpaca_comparison_train']
# settings = ['dolly_train_balent_up', 'dolly_train_balent_down', 'dolly_train_epistemic_up']
# settings = ['dolly_train_epistemic_down', 'dolly_train_aleatoric_up', 'dolly_train_aleatoric_down']
# settings = ['alpaca_train_balent_up', 'alpaca_train_balent_down', 'alpaca_train_epistemic_up']
# settings = ['alpaca_train_epistemic_down', 'alpaca_train_aleatoric_up', 'alpaca_train_aleatoric_down']
# settings = ['alpaca_train_aleatoric_down']
# settings = ['dolly_train_balent_up2', 'dolly_train_balent_down2', 'dolly_train_epistemic_up2', 'dolly_train_epistemic_down2',
# 'dolly_train_aleatoric_up2', 'dolly_train_aleatoric_down2']
# settings = ['alpaca_train_balent_up2', 'alpaca_train_balent_down2', 'alpaca_train_epistemic_up2', 'alpaca_train_epistemic_down2',
# 'alpaca_train_aleatoric_up2', 'alpaca_train_aleatoric_down2']
# settings = ['dolly_train_balent_down',]# 'alpaca_train_2']
settings = ['hh_train_random']
gpus = ['1','1', '2', '3', '4', '5']

dir_path = os.path.dirname(os.path.realpath(__file__))
my_env = os.environ

target_data = None,

import time
# time.sleep(3600)

for idx, s in enumerate(settings):
    print(s)
    setting = settings_dict[s]
    gpu = gpus[idx]
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    if setting['checkpoint_dir']=="None":
        cmd_list = ["python", os.path.join(dir_path, "src/train_bash.py"), 
                    "--stage", "rm",
                    "--model_name_or_path", setting['model_name_or_path'],
                    "--do_train",
                    "--dataset", setting['dataset'],
                    "--template", "default",
                    "--finetuning_type", "lora",
                    "--overwrite_output_dir",
                    "--lora_target", "q_proj,v_proj",
                    "--resume_lora_training", setting['resume_lora_training'],
                    # "--checkpoint_dir", setting['checkpoint_dir'],
                    "--output_dir", setting['output_dir'],
                    "--per_device_train_batch_size", "2",
                    "--gradient_accumulation_steps", "4 ",
                    "--lr_scheduler_type", "cosine",
                    "--logging_steps", "10",
                    "--save_steps", "1000",
                    "--learning_rate", setting['learning_rate'],
                    "--num_train_epochs", "1.0",
                    "--plot_loss",
                    "--fp16"
                    ]
    else:
        cmd_list = ["python", os.path.join(dir_path, "src/train_bash.py"), 
                    "--stage", "rm",
                    "--model_name_or_path", setting['model_name_or_path'],
                    "--do_train",
                    "--dataset", setting['dataset'],
                    "--template", "default",
                    "--finetuning_type", "lora",
                    "--overwrite_output_dir",
                    "--lora_target", "q_proj,v_proj",
                    "--resume_lora_training", setting['resume_lora_training'],
                    "--checkpoint_dir", setting['checkpoint_dir'],
                    "--output_dir", setting['output_dir'],
                    "--per_device_train_batch_size", "2",
                    "--gradient_accumulation_steps", "4 ",
                    "--lr_scheduler_type", "cosine",
                    "--logging_steps", "10",
                    "--save_steps", "1000",
                    "--learning_rate", setting['learning_rate'],
                    "--num_train_epochs", "1.0",
                    "--plot_loss",
                    "--fp16"
                    ]
    print(cmd_list)
    print(' '.join(cmd_list))
    p1 = subprocess.Popen(cmd_list, env=my_env)
p1.wait()
