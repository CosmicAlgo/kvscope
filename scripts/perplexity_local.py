#!/usr/bin/env python3
"""WikiText-103 perplexity baseline measurement on local RTX 4060.

Measures perplexity for models that fit in 8GB VRAM:
- Pythia-1.4B (fits easily)
- Can extend to other small models
"""
import json, os, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_local")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {"name": "EleutherAI/pythia-1.4b-deduped", "short": "pythia-1.4b"},
]

def measure_perplexity(model_name, tokenizer, model, device, context_len=2048, stride=512):
    """Measure WikiText-103 perplexity using sliding-window approach."""
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    text = "\n\n".join(dataset["text"])
    
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc  # targets include previous context
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # only compute loss on last trg_len tokens
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        if not math.isnan(neg_log_likelihood.item()) and not math.isinf(neg_log_likelihood.item()):
            nlls.append(neg_log_likelihood.item())
        
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break
        
        if begin_loc % (stride * 10) == 0:
            print(f"  Step {begin_loc}/{seq_len}, running PPL={math.exp(sum(nlls)/len(nlls)):.2f}" if nlls else "")
    
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float('inf')
    return ppl, len(nlls), seq_len

def main():
    results = []
    
    for m in MODELS:
        print(f"\n{'='*60}")
        print(f"  Measuring perplexity: {m['name']}")
        print(f"{'='*60}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(m["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            m["name"],
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        
        print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB" if torch.cuda.is_available() else "")
        
        ppl, n_chunks, total_tokens = measure_perplexity(
            m["name"], tokenizer, model, device,
            context_len=2048, stride=512
        )
        
        print(f"\n  Result: PPL = {ppl:.2f} ({n_chunks} chunks, {total_tokens} tokens)")
        
        results.append({
            "model": m["name"],
            "short_name": m["short"],
            "perplexity": round(ppl, 2),
            "n_chunks": n_chunks,
            "total_tokens": total_tokens,
            "context_len": 2048,
            "stride": 512,
            "device": str(device),
        })
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
    
    # Save
    out_path = os.path.join(RESULTS_DIR, "perplexity_baselines.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    # Summary
    print(f"\n{'Model':<30} {'PPL':>8} {'Chunks':>7} {'Tokens':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['short_name']:<30} {r['perplexity']:>8.2f} {r['n_chunks']:>7} {r['total_tokens']:>8}")

if __name__ == "__main__":
    main()
