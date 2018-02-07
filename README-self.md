## Problems met during learning t2t will be set here.
###  Add Problem
In addition to defining "Problem" class, add your "Problem" into **tensor2tensor/tensor2tensor/data_generators/all_problems.py** like this: 
``` 
from tensor2tensor.data_generators import translate_mnzh
``` 
Thus, t2t can find your self-defined "Problem".

Here, module **translate_mnzh.py** is set below data_generators. 

After running  ` t2t-datagen `, your "Problem" will be found from the output. 

---
### Train 
#### Run this script below within python environment of python3, otherwise, an error like 'InvalidArgumentError: indices[0] = -1 is not in [0, 2) will possibly come out'
```bash
#!/bin/bash
PROBLEM=translate_mnzh_bpe32k
MODEL=transformer
HPARAMS=transformer_jack

DATA_DIR=~/corpus/mn-zh/trans/mn-zh-split/t2t
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$DATA_DIR/$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=2 \
  --batch_size=2048 \
  --train_steps=2000 \
  --output_dir=$TRAIN_DIR

```
---
### About saving the trained model
   *: It is advised that flag '--export_saved_model' has been DEPRECATED. 
     Instead, we are supposed to use "serving/export.py"*
```python
flags.DEFINE_bool("export_saved_model", False,
                  "DEPRECATED - see serving/export.py.")
```
---


### Params & Flags
* Q: How many threads will be used? <br>
  [A:  If you are computing on CPU (not GPU), TensorFlow uses all available CPU cores by default.<br>
  You can limit this to N cores by adding the following lines:
  ](https://github.com/tensorflow/tensor2tensor/issues/563)
 ```python
    intra_op_parallelism_threads=N,
    inter_op_parallelism_threads=N,
```
to
```
tensor2tensor/tensor2tensor/utils/trainer_lib.py
config = tf.ConfigProto( 
allow_soft_placement=True, 
graph_options=graph_options, 
gpu_options=gpu_options, 
log_device_placement=log_device_placement) 
```
---

### T2T ( tensor2tensor ) Possible Params(Not flags)
```python
relu_dropout_broadcast_dims="",
min_length_bucket=8,
length_bucket_step=1.1,
label_smoothing=0.1,
learning_rate_warmup_steps=16000,
learning_rate_decay_scheme="noam",
filter_size=2048.0,
symbol_modality_num_shards=16,
scheduled_sampling_prob=0.0,
summarize_vars=False,
sampling_temp=1.0,
weight_decay=0.0,
prepend_mode="none",
optimizer_momentum_momentum=0.9,
initializer="uniform_unit_scaling",
learning_rate_decay_staircase=False,
symbol_modality_skip_top=False,
learning_rate_decay_rate=1.0,
use_fixed_batch_size=False,
norm_type="layer",
attention_dropout_broadcast_dims="",
optimizer="Adam",
layer_prepostprocess_dropout_broadcast_dims="",
shared_embedding_and_softmax_weights="False",
moe_k=2,
symbol_dropout=0.0,
dropout=0.2,
max_relative_position=0,
num_encoder_layers=6.0,
attention_value_channels=0.0,
multiply_embedding_mode="sqrt_depth",
compress_steps=0,
optimizer_adam_beta1=0.9,
learning_rate_cosine_cycle_steps=250000,
scheduled_sampling_gold_mixin_prob=0.5,
kernel_width=1,
target_modality="default",
eval_drop_long_sequences=False,
sampling_method="argmax",
num_heads=8.0,
eval_run_autoregressive=False,
batch_size=6250.0,
proximity_bias=False,
parameter_attention_value_channels=0,
hidden_size=512.0,
kernel_height=3,
optimizer_momentum_nesterov=False,
self_attention_type="dot_product",
pos="timing",
summarize_grads=False,
factored_logits=False,
moe_loss_coef=0.01,
learning_rate_minimum=None,
weight_noise=0.0,
parameter_attention_key_channels=0,
scheduled_sampling_warmup_steps=50000,
max_input_seq_length=0,
learning_rate_decay_steps=5000,
initializer_gain=1.0,
problem_choice="adaptive",
attention_key_channels=0.0,
min_length=0,
split_to_length=0,
daisy_chain_variables=True,
moe_num_experts=64,
max_target_seq_length=0,
max_length=256.0,
layer_postprocess_sequence="da",
optimizer_adam_beta2=0.997,
relu_dropout=0.0,
num_hidden_layers=6,
force_full_predict=False,
grad_noise_scale=0.0,
use_pad_remover=True,
optimizer_adam_epsilon=1e-09,
moe_hidden_sizes=2048.0,
input_modalities="default",
learning_rate=1.0,
clip_grad_norm=0.0,
ffn_layer="dense_relu_dense",
norm_epsilon=1e-06,
layer_prepostprocess_dropout=0.1,
num_decoder_layers=6.0,
attention_dropout=0.0,
nbr_decoder_problems=1,
no_data_parallelism=False,
layer_preprocess_sequence="n",
learning_rate_decay_scheme="noam",
learning_rate_boundaries=[0],
shared_source_target_embedding=False
```
#### T2T(tensor2tensor) Possible Flags
```python
ps_gpu=0.0,
random_seed=1234.0,
master="",
dbgprofile=False,
ps_job="/job:ps",
eval_use_test_set=False,
problems="translate_mnzh_by_32kbpe",
iterations_per_loop=100.0,
sync=False,
hparams_set="transformer_base_single_gpu",
output_dir="",
eval_early_stopping_metric="loss",
eval_early_stopping_metric_delta=0.1,
nocloud_delete_on_done="",
hparams="",
worker_id=0.0,
export_saved_model=False,
noenable_graph_rewriter="",
nohelp="",
data_dir="",
model="transformer",
keep_checkpoint_max=5,
eval_early_stopping_steps=None,
ps_replicas=0.0,
schedule="continuous_train_and_eval",
notfdbg="",
nocloud_tpu="",
noregistry_help="",
noprofile="",
enable_graph_rewriter=False,
eval_run_autoregressive=False,
worker_gpu_memory_fraction=0.95,
log_device_placement=False,
cloud_vm_name="jack-vm",
decode_hparams="",
nohelpfull="",
nogenerate_data="",
noexport_saved_model="",
noeval_use_test_set="",
nouse_tpu="",
nolog_device_placement="",
timit_paths="",
save_checkpoints_secs=0.0,
worker_job="/job:localhost",
nodbgprofile="",
keep_checkpoint_every_n_hours=10000.0,
nohelpshort="",
noeval_run_autoregressive="",
gpu_order="",
eval_steps=2000,
worker_gpu=2.0,
tpu_num_shards=8.0,
worker_replicas=1.0,
local_eval_frequency=1000.0,
cloud_tpu_name="jack-tpu",
nosync="",
parsing_path="",
hparams_range=None,
nolocally_shard_to_cpu="",
eval_early_stopping_metric_minimize=True,
tmp_dir="/tmp/t2t_datagen",
train_steps=200000,
registry_help=False,
tfdbg=False,
locally_shard_to_cpu=False
```