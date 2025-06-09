CUDA_VISIBLE_DEVICES="0" \
accelerate launch \
--main_process_port 13647 \
--num_processes 1 \
--gpu_ids 0 \
../main.py \
--input_path "C:/Users/Windows/Documents/Meri/input_data" \
--save_dir "C:/Users/Windows/Documents/Meri/latent_output" \
--train_type single \
--type_token \
--dpe \
--pos_enc \
--n_layers 2 \
--batch_size 64 \
--wandb_project_name meri_latent_extraction \
--wandb_entity_name meri_lab \
--seed 42 \
--src_data mimiciii_cv \
--mixed_precision no \
--extract_latent \
--exp_name meri_host_model \
--debug
