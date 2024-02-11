alg_name: "GRACE"
model_name: "./hugging_cache/gpt-j-6B"
device: 0

inner_params:
- transformer.h[25].mlp.fc_out.weight

edit_lr: 1.0
n_iter: 200
eps: 1.0
dist_fn: euc # euc, mmd, cos
val_init: cold # cold, warm
val_train: sgd # sgd, pert
val_reg: None # early
reg: early_stop # early_stop
replacement: replace_last # replace_last, replace_all, replace_prompt
eps_expand: coverage # , moving_avg, decay
num_pert: 8 # only matters when using perturbation training
dropout: 0.0
