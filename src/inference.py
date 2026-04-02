"""Inference avec adapteurs LoRA — SFT ou DPO."""

import sys

from mlx_lm import load, generate

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTERS = "output/sft_v2_adapters"
DPO_ADAPTERS = "output/dpo_adapters"

SYSTEM_PROMPT = (
    "Tu es un assistant specialise en IA, securite "
    "et architectures d'agents autonomes."
)


def load_model(model_name=DEFAULT_MODEL, adapter_path=SFT_ADAPTERS):
    """Charge le modele avec adapteurs LoRA (SFT ou DPO)."""
    print(f"[inference] Modele: {model_name}")
    print(f"[inference] Adapteurs: {adapter_path}")
    return load(model_name, adapter_path=adapter_path)


def ask(model, tokenizer, question, max_tokens=300):
    """Pose une question au modele."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def main():
    """CLI : python -m src.inference 'question' [sft|dpo]."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference 'question' [sft|dpo]")
        sys.exit(1)

    question = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "sft"

    adapter = SFT_ADAPTERS if mode == "sft" else DPO_ADAPTERS
    model, tokenizer = load_model(adapter_path=adapter)

    print(f"\n[Q] {question}")
    response = ask(model, tokenizer, question)
    print(f"[A] {response}")


if __name__ == "__main__":
    main()
