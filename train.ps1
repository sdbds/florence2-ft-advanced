# LoRA train script by @Akegarasu modify by @bdsqlsz

#训练模式(<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")
$train_mode = "<MORE_DETAILED_CAPTION>"

# Train data path | 设置训练用模型、图片
$pretrained_model = "microsoft/Florence-2-large" # base model path | 底模路径
$train_data_dir = "F:\florence2dataset\1_Homestay" # train dataset path | 训练数据集路径

# Train related params | 训练相关参数
$batch_size = 1 # batch size 一次性训练图片批处理数量，根据显卡质量对应调高。
$epochs = 8 # max train epoches | 最大训练 epoch

$accumulation_steps = 16 # 梯度累加数量，变相放大batchsize的倍数
$precision="fp16"
$train_split=0.8
$save_best_model=1
$output_name="Homestay"
$optimizer_accumulation_steps=1

# Learning rate | 学习率
$lr = "1e-5"
$lr_scheduler = "warmup_stable_decay" 
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_decay_steps = 2 # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
$lr_scheduler_num_cycles = 1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$lr_scheduler_timescale = 0 #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

#optimizer | 优化器
$optimizer_type = "PagedAdamW8bit"
# 可选优化器"adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增DAdaptAdam、DAdaptLion、DAdaptAdanIP，强烈推荐DAdaptAdam
# 新增优化器"Sophia"(2倍速1.7倍显存)、"Prodigy"天才优化器，可自适应Dylora
# PagedAdamW8bit、PagedLion8bit、Adan、Tiger
# AdamWScheduleFree、SGDScheduleFree
# StableAdamW、Ranger
$d_coef = "0.5" #prodigy D上升速度
$d0 = "1e-4" #dadaptation以及prodigy初始学习率
$fused_backward_pass = 0 #训练大模型float32精度专用节约显存，必须优化器adafactor或者adamw，gradient_accumulation_steps必须为1或者不开。

#wandb 日志同步
$wandb_api_key = "9c3747c46705bd779c58799295e6bb6d3da5dc98" # wandbAPI KEY，用于登录



# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$network_module = "networks.lora"
$ext_args = [System.Collections.ArrayList]::new()

if ($optimizer_type -ieq "adafactor") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("scale_parameter=False")
  [void]$ext_args.Add("warmup_init=False")
  [void]$ext_args.Add("relative_step=False")
  if ($lr_scheduler -and $lr_scheduler -ine "constant") {
    $lr_warmup_steps = 100
  }
}

if ($optimizer_type -ilike "DAdapt*") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  if ($optimizer_type -ieq "DAdaptation" -or $optimizer_type -ilike "DAdaptAdam*") {
    [void]$ext_args.Add("decouple=True")
    if ($optimizer_type -ieq "DAdaptAdam") {
      [void]$ext_args.Add("use_bias_correction=True")
    }
  }
  $lr = "1"
}

if ($optimizer_type -ieq "Lion" -or $optimizer_type -ieq "Lion8bit" -or $optimizer_type -ieq "PagedLion8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.95,.98")
}

if ($optimizer_type -ieq "AdamW8bit") {
  $optimizer_type = ""
  [void]$ext_args.Add("--use_8bit_adam")
}

if ($optimizer_type -ieq "PagedAdamW8bit" -or $optimizer_type -ieq "AdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.1")
}

if ($optimizer_type -ieq "Sophia") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SophiaH")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Prodigy") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.99")
  [void]$ext_args.Add("decouple=True")
  [void]$ext_args.Add("use_bias_correction=True")
  [void]$ext_args.Add("d_coef=$d_coef")
  if ($lr_warmup_steps) {
    [void]$ext_args.Add("safeguard_warmup=True")
  }
  if ($d0) {
    [void]$ext_args.Add("d0=$d0")
  }
  $lr = "1"
}

if ($optimizer_type -ieq "Ranger") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ieq "Adan") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ieq "StableAdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ieq "Tiger") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Tiger")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ilike "*ScheduleFree") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.08")
  [void]$ext_args.Add("weight_lr_power=0")
}

if ($accumulation_steps) {
  [void]$ext_args.Add("--accumulation_steps=$accumulation_steps")
}

if ($lr_scheduler) {
  [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($lr_scheduler_num_cycles) {
  [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}

if ($lr_warmup_steps) {
  if ($accumulation_steps) {
    $lr_warmup_steps = $lr_warmup_steps * $accumulation_steps
  }
  [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}

if ($lr_decay_steps) {
  if ($accumulation_steps) {
    $lr_decay_steps = $lr_decay_steps * $accumulation_steps
  }
  [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}

if ($lr_scheduler_timescale) {
  [void]$ext_args.Add("--lr_scheduler_timescale=$lr_scheduler_timescale")
}

if ($lr_scheduler_min_lr_ratio) {
  [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($save_best_model) {
  [void]$ext_args.Add("--save_best_model")
}

if ($precision) {
  [void]$ext_args.Add("--precision=$precision")
}

if ($output_name) {
  [void]$ext_args.Add("--output_name=$output_name")
}

if ($optimizer_accumulation_steps > 1 ) {
  [void]$ext_args.Add("--optimizer_accumulation_steps=$optimizer_accumulation_steps")
}

# run train
python "./main.py" `
  --model_dir=$pretrained_model `
  --images_dir=$train_data_dir `
  --task_type=$train_mode `
  --output_dir="./output" `
  --epochs=$epochs `
  --learning_rate=$lr `
  --batch_size=$batch_size `
  $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;