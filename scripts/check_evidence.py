import json
for name in ['gptoss', 'nemotron', 'spec_decode']:
    d = json.load(open(f'final_data/results_v2/{name}_profile.json', encoding='utf-8'))
    pp = d['per_prompt'][0]
    ld = pp.get('leak_detection', {})
    print(f'\n=== {name} ===')
    for f in ld.get('findings', []):
        det = f['detector']
        ev = f.get('evidence', {})
        keys_of_interest = ['bytes_per_token_per_layer','n_active_layers','cv','mean_kb',
                           'total_bytes_at_end','peak_seq_len','category']
        filtered = {k: ev[k] for k in keys_of_interest if k in ev}
        if filtered:
            print(f'  {det}: {filtered}')
    detectors = [f['detector'] for f in ld.get('findings', [])]
    print(f'  All detectors: {detectors}')
