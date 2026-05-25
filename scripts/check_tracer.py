import json
for name in ['gptoss', 'nemotron', 'spec_decode']:
    d = json.load(open(f'final_data/results_v2/{name}_profile.json', encoding='utf-8'))
    pp = d['per_prompt']
    # Find a prompt with actual tokens
    for p in pp:
        if p.get('actual_new_tokens', 0) > 100:
            tracer = p.get('tracer', {})
            kv = tracer.get('kv_cache', {})
            print(f'\n=== {name} (prompt {p["prompt_idx"]}, {p["actual_new_tokens"]} tok) ===')
            print(f'tracer keys: {list(tracer.keys())}')
            print(f'kv_cache keys: {list(kv.keys()) if kv else "EMPTY"}')
            if kv:
                # Print first few items
                for k, v in kv.items():
                    if isinstance(v, list) and len(v) > 0:
                        print(f'  {k}: list[{len(v)}], first={v[0] if len(v)==1 else v[0]}')
                    elif isinstance(v, dict):
                        print(f'  {k}: dict keys={list(v.keys())[:10]}')
                    else:
                        print(f'  {k}: {v}')
            break
    else:
        # No long prompt found, show first prompt
        p = pp[0]
        tracer = p.get('tracer', {})
        print(f'\n=== {name} (prompt {p["prompt_idx"]}, {p.get("actual_new_tokens",0)} tok) ===')
        print(f'tracer keys: {list(tracer.keys())}')
        kv = tracer.get('kv_cache', {})
        print(f'kv_cache: {kv}')
