"""
MODEL-ONLY INFERENCE
LoRA fine-tuned TinyLlama
NO RAG, NO FAISS, NO PDFs
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# =========================
# MODEL CONFIG
# =========================
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_ID = "Akankshamulik/tinyllama-lora-training"

device = "cpu"  # change to "cuda" if GPU available

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_ID,
    trust_remote_code=True
)

# =========================
# LOAD BASE MODEL
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None
).to(device)

# =========================
# LOAD LoRA ADAPTER
# =========================
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID
)

model.eval()

# =========================
# GENERATION FUNCTION
# =========================
def generate_answer(question: str) -> str:
    prompt = f"""
You are an NCERT teacher.
Answer the question in 1–2 lines.
Give only the definition.

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.7
        )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # clean output
    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    return text.strip().split("\n")[0]


# =========================
# RUN LOOP
# =========================
print("\n✅ MODEL-ONLY SYSTEM READY")
print("Type a question (type 'exit' to quit)\n")

while True:
    q = input("🧠 Question: ")

    if q.lower() in ["exit", "quit"]:
        print("👋 Bye")
        break

    ans = generate_answer(q)
    print("\n🔹 Answer:\n", ans)
    print("\n" + "-" * 50)