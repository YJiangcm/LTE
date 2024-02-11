alg_name: "MEMIT"
model_name: THUDM/chatglm2-6b
stats_dir: "./data/stats"
device: 0
layers: [13, 14, 15, 16, 17]
clamp_norm_factor: 4
layer_selection: "all"
fact_token: "subject_last"
#the place of token
v_num_grad_steps: 25
#the times of memit

v_lr: 5e-1
v_loss_layer: 27
v_weight_decay: 1e-3
kl_factor: 0.0625
mom2_adjustment: true
mom2_update_weight: 15000

rewrite_module_tmp: "transformer.encoder.layers.{}.mlp.dense_4h_to_h"
layer_module_tmp: "transformer.encoder.layers.{}"
mlp_module_tmp: "transformer.encoder.layers.{}.mlp"
attn_module_tmp: "transformer.encoder.layers.{}.self_attention"
ln_f_module: "transformer.encoder.final_layernorm"
lm_head_module: "transformer.output_layer"
#serach layer
mom2_dataset: "wikipedia"
mom2_n_samples: 100000
mom2_dtype: "float32"
