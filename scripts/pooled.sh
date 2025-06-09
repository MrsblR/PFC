CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--main_process_port 13647 \
--num_processes 4 \
--gpu_ids 0,1,2,3 \
../main.py \
--input_path "C:/Users/Windows/Documents/Meri/input_data" \
--save_dir "C:/Users/Windows/Documents/Meri/output_results" \
--train_type pooled \
--type_token \
--dpe \
--pos_enc \
--n_layers 2 \
--batch_size 64 \
--wandb_project_name meri_pooling_exp \
--wandb_entity_name meri_lab \
--seed 42 \
--src_data mimiciii_cv mimiciii_mv mimiciv eicu_south eicu_west \
--mixed_precision bf16
