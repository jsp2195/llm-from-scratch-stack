#!/usr/bin/env python
import json
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location='cpu')
out = sys.argv[2]
torch.save(ckpt['model'], out)
with open(out + '.manifest.json', 'w', encoding='utf-8') as f:
    json.dump({'source': sys.argv[1], 'keys': list(ckpt.keys())}, f, indent=2)
