###################################################
## Full batch is written by Jae Oh
###################################################
import os
import subprocess

settings_dict = {}
settings_dict['dolly_regular_train'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_regular_train_mistral7b'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'mistralai/Mistral-7B-v0.1',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4_mistral7b',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_regular_train_13b'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-13b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4_13b',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_regular_train_ep5'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_ep0_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_down_ep5'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_ep0_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_down2'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down2_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_up2'] = {
    'dataset': 'dolly_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_lr4',
    'resume_lora_training': "True", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up2_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_down'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_up'] = {
    'dataset': 'dolly_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up_lr4',
    'lr_scheduler_type': 'cosine',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_down_nosc'] = {
    'dataset': 'dolly_train_balent_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_up_nosc'] = {
    'dataset': 'dolly_train_balent_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_nosc'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_epistemic_down_nosc'] = {
    'dataset': 'dolly_train_epistemic_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_down_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_epistemic_up_nosc'] = {
    'dataset': 'dolly_train_epistemic_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_up_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_aleatoric_down_nosc'] = {
    'dataset': 'dolly_train_aleatoric_down',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_down_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['dolly_train_aleatoric_up_nosc'] = {
    'dataset': 'dolly_train_aleatoric_up',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_up_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_mixed_nosc'] = {
    'dataset': 'dolly_train_balent_mixed',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_mixed2_nosc'] = {
    'dataset': 'dolly_train_balent_mixed2',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed2_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_mixed3_nosc'] = {
    'dataset': 'dolly_train_balent_mixed3',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed3_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['dolly_train_balent_mixed4_nosc'] = {
    'dataset': 'dolly_train_balent_mixed4',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed4_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['dolly_train_balent_mixed5_nosc'] = {
    'dataset': 'dolly_train_balent_mixed5',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed5_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['dolly_train_balent_mixed6_nosc'] = {
    'dataset': 'dolly_train_balent_mixed6',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed6_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train'] = {
    'dataset': 'combined_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/combined_train_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling'] = {
    'dataset': 'combined_train_weighted_sampling',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/combined_train_weighted_sampling_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['combined_train_loss9'] = {
    'dataset': 'combined_train',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/combined_train_loss9_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['combined_train_weighted_sampling_loss9'] = {
    'dataset': 'combined_train_weighted_sampling',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/balanced_entropy_rm_splits/combined_train_weighted_sampling_loss9_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

settings_dict['combined_train_weighted_sampling1'] = {
    'dataset': 'combined_train_weighted_sampling1',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling1_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling2'] = {
    'dataset': 'combined_train_weighted_sampling2',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling2_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling3'] = {
    'dataset': 'combined_train_weighted_sampling3',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling3_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling4'] = {
    'dataset': 'combined_train_weighted_sampling4',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling4_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling5'] = {
    'dataset': 'combined_train_weighted_sampling5',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling5_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_weighted_sampling6'] = {
    'dataset': 'combined_train_weighted_sampling6',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_weighted_sampling6_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling1'] = {
    'dataset': 'combined_train_random_sampling1',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling1_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling2'] = {
    'dataset': 'combined_train_random_sampling2',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling2_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling3'] = {
    'dataset': 'combined_train_random_sampling3',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling3_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling4'] = {
    'dataset': 'combined_train_random_sampling4',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling4_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling5'] = {
    'dataset': 'combined_train_random_sampling5',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling5_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}
settings_dict['combined_train_random_sampling6'] = {
    'dataset': 'combined_train_random_sampling6',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': "None",
    'resume_lora_training': "False", 
    'output_dir': './results/icml_rm/combined_train_random_sampling6_nosc_lr4',
    'lr_scheduler_type': 'constant',
    'learning_rate': '1e-4',
    'num_train_epochs': '1.0'
}

# settings = ['combined_train_random_sampling6', 'combined_train_random_sampling5',
# 'combined_train_random_sampling4', 'combined_train_random_sampling3', 'combined_train_random_sampling2']


# settings = ['combined_train_weighted_sampling6', 'combined_train_weighted_sampling5',
# 'combined_train_weighted_sampling4', 'combined_train_weighted_sampling3', 'combined_train_weighted_sampling2']

#['dolly_train_balent_mixed6_nosc']

#settings = ['dolly_train_balent_mixed4_nosc',]#'dolly_train_balent_mixed3_nosc']
#['combined_train_loss9', 'combined_train_weighted_sampling_loss9']
#['combined_train', 'combined_train_weighted_sampling']
#['dolly_train_balent_mixed6_nosc', 'dolly_train_balent_mixed5_nosc','dolly_train_balent_mixed4_nosc', 'dolly_train_balent_mixed3_nosc','dolly_train_balent_mixed2_nosc', 'dolly_train_balent_mixed_nosc',]
#['dolly_train_balent_down_nosc', 'dolly_train_balent_up_nosc', 'dolly_train_epistemic_up_nosc', 'dolly_train_epistemic_down_nosc','dolly_train_aleatoric_up_nosc', 'dolly_train_aleatoric_down_nosc',]#'dolly_train_balent_down_nosc']#'dolly_train_balent_mixed6_nosc']
#['dolly_train_balent_up_nosc', 'dolly_train_epistemic_up_nosc', 'dolly_train_aleatoric_up_nosc','dolly_train_epistemic_down_nosc']
#'dolly_train_balent_mixed4_nosc', 'dolly_train_balent_mixed3_nosc',]# 'dolly_train_balent_mixed3_nosc', 'dolly_train_balent_mixed4_nosc','dolly_train_balent_mixed5_nosc', 'dolly_train_balent_mixed_nosc']#]
#['dolly_train_balent_down_nosc', 'dolly_train_balent_up_nosc', 'dolly_train_epistemic_up_nosc', 'dolly_train_epistemic_down_nosc','dolly_train_aleatoric_up_nosc', 'dolly_train_aleatoric_down_nosc',]#'dolly_train_balent_down_nosc']#'dolly_train_balent_mixed6_nosc']
#['dolly_train_balent_up_nosc', 'dolly_train_epistemic_up_nosc', 'dolly_train_aleatoric_up_nosc','dolly_train_epistemic_down_nosc']
#[ 'dolly_train_balent_mixed_nosc', 'dolly_train_balent_mixed2_nosc', 'dolly_train_balent_mixed3_nosc', 'dolly_train_balent_mixed4_nosc','dolly_train_balent_mixed5_nosc', 'dolly_train_balent_mixed6_nosc']#]
#,
# ['dolly_train_balent_down_nosc', 'dolly_train_epistemic_down_nosc', 'dolly_train_aleatoric_down_nosc',
#'dolly_train_nosc'  ]#'']
#['dolly_train_epistemic_down_nosc', 'dolly_train_aleatoric_down_nosc']
#'dolly_train_balent_down_nosc', 'dolly_train_balent_up_nosc', 'dolly_train_balent_down', 'dolly_train_balent_up']#, 'dolly_train_balent_down_ep5']
settings = ['dolly_regular_train_mistral7b']#'dolly_regular_train_13b']
gpus = [  '5','4','3', '2','1', '0',]

dir_path = os.path.dirname(os.path.realpath(__file__))
my_env = os.environ

target_data = None,

import time
# time.sleep(3600*6)

# checkpoints = {}
# num_epochs = 5
processes = []
# for i in range(num_epochs):

for idx, s in enumerate(settings):
    # print(s)
    # print('current_checkpoint')
    # print(checkpoints)
    setting = settings_dict[s]
    # if s in checkpoints.keys():
    #     setting['checkpoint_dir'] = checkpoints[s]
    #     setting['resume_lora_training'] = 'True'
    # setting['output_dir'] = setting['output_dir'].replace("ep"+str(i), "ep"+str(i+1))
    # checkpoints[s] = setting['output_dir']
    # print('next_checkpoint')
    # print(checkpoints)

    gpu = gpus[idx]
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    if setting['checkpoint_dir']=="None":
        cmd_list = ["python", os.path.join(dir_path, "src/train_bash.py"), 
                    "--stage", "rm",
                    "--model_name_or_path", setting['model_name_or_path'],
                    "--do_train",
                    # "--use_balent_loss2",
                    "--dataset", setting['dataset'],
                    "--template", "default",
                    "--finetuning_type", "lora",
                    "--overwrite_output_dir",
                    "--lora_target", "q_proj,v_proj",
                    "--resume_lora_training", setting['resume_lora_training'],
                    # "--checkpoint_dir", setting['checkpoint_dir'],
                    "--output_dir", setting['output_dir'],
                    "--per_device_train_batch_size", "1",
                    "--gradient_accumulation_steps", "4 ",
                    "--lr_scheduler_type", setting['lr_scheduler_type'],
                    "--logging_steps", "10",
                    "--save_steps", "1000",
                    "--learning_rate", setting['learning_rate'],
                    "--num_train_epochs", setting['num_train_epochs'],
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
                    "--per_device_train_batch_size", "1",
                    "--gradient_accumulation_steps", "4 ",
                    "--lr_scheduler_type", setting['lr_scheduler_type'],
                    "--logging_steps", setting['num_train_epochs'],
                    "--save_steps", "1000",
                    "--learning_rate", setting['learning_rate'],
                    "--num_train_epochs", setting['num_train_epochs'],
                    "--plot_loss",
                    "--fp16"
                    ]
    print(cmd_list)
    print(' '.join(cmd_list))
    processes.append(subprocess.Popen(cmd_list, env=my_env))
for p in processes:
    p.wait()
processes = []

settings_dict['dolly_train_13b'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'meta-llama/Llama-2-13b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4_13b',#settings_dict['dolly_regular_train_ep5']['output_dir'],
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_train_13b',
}

settings_dict['dolly_train_mistral7b'] = {
    'dataset': 'dolly_train',
    'model_name_or_path': 'mistralai/Mistral-7B-v0.1',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/train_dolly_lr4_mistral7b',#settings_dict['dolly_regular_train_ep5']['output_dir'],
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_train_mistral7b',
}

settings_dict['dolly_test_ep5'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_ep3_lr4',#settings_dict['dolly_regular_train_ep5']['output_dir'],
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_ep3_lr4',
}

settings_dict['dolly_test_balent_down_ep5'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_ep3_lr4',#settings_dict['dolly_train_balent_down_ep5']['output_dir'],
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_down_ep3_lr4',
}

settings_dict['dolly_test_balent_up'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_up_lr4',
}

settings_dict['dolly_test_balent_down'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_down_lr4',
}

settings_dict['dolly_test_balent_up_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_up_nosc_lr4',
}

settings_dict['dolly_test_balent_down_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_down_nosc_lr4',
}

settings_dict['dolly_test_balent_down2'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down2_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_down2_lr4',
}

settings_dict['dolly_test_balent_up2'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_up2_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_up2_lr4',
}

settings_dict['dolly_test_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_nosc_lr4',
}

settings_dict['dolly_test_epistemic_down_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_epistemic_down_nosc_lr4',
}
settings_dict['dolly_test_epistemic_up_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_up_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_epistemic_up_nosc_lr4',
}

settings_dict['dolly_test_aleatoric_down_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_aleatoric_down_nosc_lr4',
}
settings_dict['dolly_test_aleatoric_up_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_up_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_aleatoric_up_nosc_lr4',
}

settings_dict['dolly2alpaca_test_balent_down_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly2alpaca_test_balent_down_nosc_lr4',
}

settings_dict['dolly2alpaca_test_epistemic_down_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly2alpaca_test_epistemic_down_nosc_lr4',
}
settings_dict['dolly2alpaca_test_epistemic_up_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_epistemic_up_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly2alpaca_test_epistemic_up_nosc_lr4',
}

settings_dict['alpaca_test_aleatoric_down_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_down_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_alpaca_test_aleatoric_down_nosc_lr4',
}
settings_dict['alpaca_test_aleatoric_up_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_aleatoric_up_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_alpaca_test_aleatoric_up_nosc_lr4',
}

settings_dict['dolly_test_balent_mixed_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed_nosc_lr4',
}
settings_dict['dolly2alpaca_test_balent_mixed_nosc'] = {
    'dataset': 'alpaca_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly2alpaca_test_balent_mixed_nosc_lr4',
}

settings_dict['dolly_test_balent_mixed2_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed2_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed2_nosc_lr4',
}
settings_dict['dolly_test_balent_mixed3_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed3_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed3_nosc_lr4',
}
settings_dict['dolly_test_balent_mixed4_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed4_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed4_nosc_lr4',
}
settings_dict['dolly_test_balent_mixed5_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed5_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed5_nosc_lr4',
}
settings_dict['dolly_test_balent_mixed6_nosc'] = {
    'dataset': 'dolly_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/dolly_train_balent_mixed6_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_dolly_test_balent_mixed6_nosc_lr4',
}

settings_dict['combined_test_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/combined_train_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_combined_test_nosc_lr4',
}
settings_dict['combined_test_weighted_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/combined_train_weighted_sampling_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_combined_test_weighted_sampling_nosc_lr4',
}

settings_dict['combined_test_loss9_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/combined_train_loss9_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_combined_test_loss9_nosc_lr4',
}
settings_dict['combined_test_weighted_loss9_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/balanced_entropy_rm_splits/combined_train_weighted_sampling_loss9_nosc_lr4',
    'output_dir': './results/balanced_entropy_rm_splits/predict_combined_test_weighted_sampling_loss9_nosc_lr4',
}
settings_dict['combined_test_weighted1_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling1_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling1_nosc_lr4',
}
settings_dict['combined_test_weighted2_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling2_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling2_nosc_lr4',
}
settings_dict['combined_test_weighted3_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling3_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling3_nosc_lr4',
}
settings_dict['combined_test_weighted4_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling4_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling4_nosc_lr4',
}
settings_dict['combined_test_weighted5_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling5_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling5_nosc_lr4',
}
settings_dict['combined_test_weighted6_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_weighted_sampling6_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_weighted_sampling6_nosc_lr4',
}
settings_dict['combined_test_random1_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling1_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling1_nosc_lr4',
}
settings_dict['combined_test_random2_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling2_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling2_nosc_lr4',
}
settings_dict['combined_test_random3_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling3_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling3_nosc_lr4',
}
settings_dict['combined_test_random4_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling4_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling4_nosc_lr4',
}
settings_dict['combined_test_random5_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling5_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling5_nosc_lr4',
}
settings_dict['combined_test_random6_nosc'] = {
    'dataset': 'combined_test',
    'model_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
    'checkpoint_dir': './results/icml_rm/combined_train_random_sampling6_nosc_lr4',
    'output_dir': './results/icml_rm/predict_combined_test_random_sampling6_nosc_lr4',
}
#
# settings = ['combined_test_random6_nosc', 'combined_test_random5_nosc', 
# 'combined_test_random4_nosc', 'combined_test_random3_nosc', 'combined_test_random2_nosc']
# settings = ['combined_test_weighted6_nosc', 'combined_test_weighted5_nosc', 
# 'combined_test_weighted4_nosc', 'combined_test_weighted3_nosc', 'combined_test_weighted2_nosc']
#settings =  ['dolly_test_balent_mixed4_nosc',]# 'dolly_test_balent_mixed3_nosc']
#['combined_test_loss9_nosc' ,'combined_test_weighted_loss9_nosc', ]
#'combined_test_nosc', 'combined_test_weighted_nosc',]
#  # ['combined_test_loss9_nosc', 'combined_test_weighted_loss9_nosc']
# ['combined_test_nosc', 'combined_test_weighted_nosc']
#['dolly_test_balent_mixed6_nosc', 'dolly_test_balent_mixed5_nosc','dolly_test_balent_mixed4_nosc', 'dolly_test_balent_mixed3_nosc', 'dolly_test_balent_mixed2_nosc', 'dolly_test_balent_mixed_nosc',]
#['dolly_test_balent_down_nosc', 'dolly_test_balent_up_nosc', 'dolly_test_epistemic_up_nosc', 'dolly_test_epistemic_down_nosc','dolly_test_aleatoric_up_nosc','dolly_test_aleatoric_down_nosc',]#'dolly_test_balent_down_nosc']# 'dolly_test_balent_mixed6_nosc']
# 'dolly_test_balent_mixed4_nosc', 'dolly_test_balent_mixed3_nosc',]#
# 'dolly_test_balent_mixed3_nosc', 'dolly_test_balent_mixed4_nosc','dolly_test_balent_mixed5_nosc', 'dolly_test_balent_mixed6_nosc']#]
#['dolly_test_balent_down_nosc', 'dolly_test_balent_up_nosc', 'dolly_test_epistemic_up_nosc', 'dolly_test_epistemic_down_nosc','dolly_test_aleatoric_up_nosc','dolly_test_aleatoric_down_nosc',]#'dolly_test_balent_down_nosc']# 'dolly_test_balent_mixed6_nosc']
# ['dolly_test_balent_up_nosc', 'dolly_test_epistemic_up_nosc', 'dolly_test_aleatoric_up_nosc','dolly_test_epistemic_down_nosc']
#[ 'dolly_test_balent_mixed_nosc', 'dolly_test_balent_mixed2_nosc', 'dolly_test_balent_mixed3_nosc', 'dolly_test_balent_mixed4_nosc','dolly_test_balent_mixed5_nosc', 'dolly_test_balent_mixed6_nosc']#]
#
#['dolly_test_balent_down_nosc', 'dolly_test_epistemic_down_nosc', 'dolly_test_aleatoric_down_nosc', 'dolly_test_nosc']
#'dolly2alpaca_test_balent_down_nosc' ]
    #'dolly_test_balent_mixed_nosc', 'alpaca_test_balent_mixed_nosc' ]
#['dolly_test_epistemic_down_nosc', 'dolly_test_aleatoric_down_nosc', 'alpaca_test_balent_down_nosc',
#'alpaca_test_epistemic_down_nosc', 'alpaca_test_aleatoric_down_nosc']
#'dolly_test_balent_down_nosc','dolly_test_balent_up_nosc', 'dolly_test_balent_down', 'dolly_test_balent_up']# 'dolly_test_balent_down_ep5'] #
settings = ['dolly_train_mistral7b']#'dolly_train_13b']
gpus = [ '0','4', '3', '2', '1', '0']

for idx, s in enumerate(settings):
    print(s)
    setting = settings_dict[s]
    gpu = gpus[idx]
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    cmd_list = ["python", os.path.join(dir_path, "src/train_bash.py"), 
                "--stage", "rm",
                "--model_name_or_path", setting['model_name_or_path'],
                "--do_predict",
                "--do_sample", "False",
                "--dataset", setting['dataset'],
                "--template", "default",
                "--finetuning_type", "lora",
                "--max_source_length", "4096",
                "--overwrite_output_dir",
                "--checkpoint_dir", setting['checkpoint_dir'],
                "--output_dir", setting['output_dir'],
                "--per_device_eval_batch_size", "2",
                "--max_samples", "10000000",
                ]
    print(cmd_list)
    p1 = subprocess.Popen(cmd_list, env=my_env)
p1.wait()