## Problems met during learning t2t will be set here.
####  Add Problem
In addition to defining "Problem" class, add your "Problem" into **tensor2tensor/tensor2tensor/data_generators/all_problems.py** like this: 
``` 
from tensor2tensor.data_generators import translate_mnzh
``` 
Thus, t2t can find your self-defined "Problem".

Here, module **translate_mnzh.py** is set below data_generators. 

After running  ` t2t-datagen `, your "Problem" will be found from the output. 

#### Possible Params (did not take into account the flags)
kernel_height=3,<br>
pos="timing",<br>
sampling_temp=1.0,<br>
num_encoder_layers=6,<br>
learning_rate=1.0,<br>
daisy_chain_variables=True,<br>
num_hidden_layers=6,<br>
self_attention_type="dot_product",<br>
eval_run_autoregressive=False,<br>
compress_steps=0,<br>
proximity_bias=False,<br>
moe_k=2,<br>
clip_grad_norm=0.0,<br>
layer_postprocess_sequence="da",<br>
learning_rate_decay_scheme="noam",<br>
scheduled_sampling_prob=0.0,<br>
weight_noise=0.0,<br>
initializer_gain=1.0,<br>
shared_embedding_and_softmax_weights=False,<br>
use_fixed_batch_size=False,<br>
prepend_mode="none",<br>
optimizer_adam_beta2=0.997,<br>
moe_loss_coef=0.01,<br>
layer_prepostprocess_dropout_broadcast_dims="",<br>
optimizer="Adam",<br>
initializer="uniform_unit_scaling",<br>
hidden_size=512,<br>
force_full_predict=False,<br>
learning_rate_decay_rate=1.0,<br>
symbol_modality_skip_top=False,<br>
sampling_method="argmax",<br>
summarize_vars=False,<br>
num_decoder_layers=6,<br>
grad_noise_scale=0.0,<br>
max_target_seq_length=0,<br>
max_length=256,<br>
split_to_length=0,<br>
parameter_attention_key_channels=0,<br>
symbol_modality_num_shards=16,<br>
attention_key_channels=0,<br>
layer_prepostprocess_dropout=0.1,<br>
input_modalities="default",<br>
label_smoothing=0.1,<br>
learning_rate_decay_steps=5000,<br>
learning_rate_minimum=None,<br>
layer_preprocess_sequence="n",<br>
learning_rate_decay_staircase=False,<br>
optimizer_adam_beta1=0.9,<br>
attention_dropout=0.0,<br>
min_length_bucket=8,<br>
summarize_grads=False,<br>
dropout=0.2,<br>
moe_hidden_sizes=2048,<br>
learning_rate_cosine_cycle_steps=250000,<br>
batch_size=6250,<br>
eval_drop_long_sequences=False,<br>
attention_value_channels=0,<br>
norm_epsilon=1e-06,<br>
nbr_decoder_problems=1,<br>
relu_dropout=0.0,<br>
scheduled_sampling_warmup_steps=50000,<br>
no_data_parallelism=False,<br>
norm_type="layer",<br>
moe_num_experts=64,<br>
relu_dropout_broadcast_dims="",<br>
target_modality="default",<br>
weight_decay=0.0,<br>
max_input_seq_length=0,<br>
optimizer_momentum_nesterov=False,<br>
factored_logits=False,<br>
min_length=0,<br>
optimizer_momentum_momentum=0.9,<br>
parameter_attention_value_channels=0,<br>
symbol_dropout=0.0,<br>
kernel_width=1,<br>
optimizer_adam_epsilon=1e-09,<br>
use_pad_remover=True,<br>
learning_rate_warmup_steps=16000,<br>
scheduled_sampling_gold_mixin_prob=0.5,<br>
problem_choice="adaptive",<br>
multiply_embedding_mode="sqrt_depth",<br>
num_heads=8,<br>
filter_size=2048,<br>
ffn_layer="dense_relu_dense",<br>
max_relative_position=0,<br>
attention_dropout_broadcast_dims="",<br>
length_bucket_step=1.1,<br>
