# -*- coding: utf-8 -*-

import sys
import json
from tokenizer import build_tokenizer

### read tokenizer
tok = {'mode': 'conservative', 'vocabulary': ''}
if len(sys.argv) > 1:
    with open(sys.argv[1]) as jsonfile: 
        tok = json.load(jsonfile)

tokenizer = build_tokenizer(tok)

for line in sys.stdin:
    line, _ = tokenizer.tokenize(str(line.strip('\n')))
    print("{}".format(" ".join(line)))
