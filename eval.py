import os, math, random, csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from leanquant import LeanQuantModelForCausalLM

# ---- helpers ----
@torch.no_grad()
def evaluate_ppl_and_acc(model, tokenizer, dataset_name, device="cuda:0", block_size=2048, batch_size=2, max_batches=None):
    print(f"Evaluating Perplexity + Accuracy on {dataset_name}...")
    texts = []
    if dataset_name == "wikitext-2-raw-v1":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    elif dataset_name == "allenai/c4":
        try:
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            for i, ex in enumerate(ds):
                if i >= 1000:  # subsample
                    break
                t = ex.get("text") or ex.get("document")
                if isinstance(t, str) and t.strip():
                    texts.append(t)
        except Exception as e:
            print(f"Error loading C4: {e}")
            return None
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    if not texts:
        print("No texts available, skip.")
        return None

    enc = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    n_blk = input_ids.size(1) // block_size
    if n_blk == 0:
        print(f"Not enough tokens for block_size {block_size}")
        return None
    input_ids = input_ids[:, : n_blk * block_size].view(n_blk, block_size)
    attention_mask = torch.ones_like(input_ids)

    total_tokens = total_correct = 0
    total_nll = 0.0

    N = input_ids.size(0)
    n_batches = min((N + batch_size - 1) // batch_size, max_batches or 10**9)

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None and hasattr(model, "lm_head"):
        vocab_size = getattr(model.lm_head, "out_features", None)
    if vocab_size is None:
        raise AttributeError("Cannot determine vocab_size")

    for b in tqdm(range(n_batches), desc=f"Eval on {dataset_name}"):
        s, e = b * batch_size, min((b + 1) * batch_size, N)
        batch_inp = input_ids[s:e].to(device)
        batch_att = attention_mask[s:e].to(device)

        outputs = model(input_ids=batch_inp, attention_mask=batch_att)
        logits = outputs.logits  # [B,T,V]

        shift_logits = logits[:, :-1, :].to(torch.float32)
        shift_labels = batch_inp[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=pad_id,
            reduction="sum"
        )
        preds = shift_logits.argmax(dim=-1)
        mask = (shift_labels != pad_id)

        total_nll += loss.item()
        total_correct += (preds[mask] == shift_labels[mask]).sum().item()
        total_tokens += mask.sum().item()

    ppl = math.exp(total_nll / max(1, total_tokens))
    acc = total_correct / max(1, total_tokens)
    return {"perplexity": ppl, "accuracy": acc, "tokens": total_tokens}


def main():
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256
    device = 'cuda:0'

    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = LeanQuantModelForCausalLM.from_pretrained(
        base_model_name,
        "./llama3.1.8b.4bit.fin.0.1.0.7.safetensors",
        bits=4,
        device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    DATASETS_TO_EVALUATE = ["wikitext-2-raw-v1", "allenai/c4"]

    results = []

    for ds_name in DATASETS_TO_EVALUATE:
        metrics = evaluate_ppl_and_acc(model, tokenizer, ds_name, device)
        if metrics:
            print(f"[{ds_name}] PPL={metrics['perplexity']:.3f}, Acc={metrics['accuracy']:.4f}, Tokens={metrics['tokens']}")
            results.append((ds_name, metrics))

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Metric", "Dataset", "Value"])
        id_counter = 0
        for ds_name, m in results:
            writer.writerow([id_counter, "Perplexity", ds_name, round(m["perplexity"], 3)]); id_counter += 1
            writer.writerow([id_counter, "Accuracy", ds_name, round(m["accuracy"], 4)]); id_counter += 1


if __name__ == '__main__':
    main()
