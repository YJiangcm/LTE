# Learning to Edit: Aligning LLM Behavior with Knowledge Editing


The training data of LTE is avaliable [here](https://huggingface.co/datasets/YuxinJiang/LTE_train_data).

The code for fine-tuning of LLaMA2-Chat-7B is based on [FastChat](https://github.com/lm-sys/FastChat).

```bash
cd LTE/
sh FastChat/ft_train.sh
```

```bash
cd LTE/
sh FastChat/lora_train.sh
```

Fine-tuning Qwen-Chat-7B https://github.com/QwenLM/Qwen

```bash
cd LTE/
sh Qwen/finetune/finetune_ds.sh
```

```bash
cd LTE/
sh Qwen/finetune/finetune_lora_single_gpu.sh
```


```bash
cd LTE/
sh EasyEdit/run_lte_llama.sh
```

```bash
cd LTE/
sh EasyEdit/run_lte_qwen.sh
```
