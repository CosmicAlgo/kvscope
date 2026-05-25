import json, os
D = os.path.join(os.path.dirname(__file__), "..", "configs_check", "results_v2", "model_configs")
for f in sorted(os.listdir(D)):
    c = json.load(open(os.path.join(D, f)))
    arch = c.get("architectures", c.get("model_type", "?"))
    layers = c.get("num_hidden_layers", "?")
    heads = c.get("num_attention_heads", "?")
    kv_heads = c.get("num_key_value_heads", "?")
    hidden = c.get("hidden_size", "?")
    print(f"{f:25s}  arch={str(arch):40s}  layers={layers:>3}  heads={heads:>3}  kv_heads={kv_heads:>3}  hidden={hidden}")
