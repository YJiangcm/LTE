# Learning to Edit: Aligning LLM Behavior with Knowledge Editing

## How to implement

### Requirements
**Note: Please use Python 3.10+ for LTE.**  To get started, simply install conda and run:
```
conda create -n LTE python=3.10
conda activate LTE
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 1. Alignment Phrase
The training data of LTE is avaliable [here](https://huggingface.co/datasets/YuxinJiang/LTE_train_data).

The code for fine-tuning of LLaMA2-Chat-7B is based on [FastChat](https://github.com/lm-sys/FastChat).

```bash
cd LTE/
bash FastChat/ft_train.sh
```

```bash
cd LTE/
bash FastChat/lora_train.sh
```
The code for fine-tuning of Qwen-Chat-7B is based on [Qwen](https://github.com/QwenLM/Qwen).

```bash
cd LTE/
bash Qwen/finetune/finetune_ds.sh
```

```bash
cd LTE/
bash Qwen/finetune/finetune_lora_single_gpu.sh
```

### 2. Inference Phrase
The evaluation of our proposed LTE is based on [EasyEdit](https://github.com/zjunlp/EasyEdit).

Please run the following command for experiments of LLaMA2-Chat-7B:
```bash
cd LTE/
bash EasyEdit/run_lte_llama.sh
```

Please run the following command for experiments of Qwen-Chat-7B:
```bash
cd LTE/
bash EasyEdit/run_lte_qwen.sh
```
