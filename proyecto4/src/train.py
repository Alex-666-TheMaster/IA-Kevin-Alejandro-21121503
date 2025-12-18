import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from huggingface_hub import login

# Cargar entorno
load_dotenv()
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


# Configuración
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
NEW_MODEL = "llama-3.2-1b-tutor-algoritmos"
DATASET_FILE = "data/processed/train_dataset.jsonl"

# Parámetros QLoRA
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05

# Parámetros de bitsandbytes (4-bit)
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Parámetros de entrenamiento
output_dir = "./results"
num_train_epochs = 10
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4 
gradient_checkpointing = True 
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25

# Cargar dataset
dataset = load_dataset('json', data_files=DATASET_FILE, split="train")

# Configurar cuantización
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Cargar modelo base
print(f"Cargando modelo base: {MODEL_NAME}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    print("Asegúrate de haber hecho login con 'huggingface-cli login' y tener acceso a Llama 3.2")
    exit(1)

# Cargar tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix para fp16

# Configuración PEFT
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Target modules para Llama
)

# Configuración SFT (Reemplaza TrainingArguments para incluir max_seq_length en TRL moderno)
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="none",
    dataset_text_field="text"
)

# Entrenador
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config
)

# Entrenar
# Verificar checkpoints existentes para reanudar
last_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Ordenar por número de paso
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        last_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f"Reanudando entrenamiento desde: {last_checkpoint}")

print("Iniciando entrenamiento...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Guardar modelo entrenado
print(f"Guardando modelo en {NEW_MODEL}...")
trainer.model.save_pretrained(NEW_MODEL)
tokenizer.save_pretrained(NEW_MODEL)

print("¡Entrenamiento completado!")
