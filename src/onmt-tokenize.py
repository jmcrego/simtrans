# -*- coding: utf-8 -*-

import sys
import json
from tokenizer import build_tokenizer

fjson = None
case = False
tokopts = {'mode': 'aggressive', 'vocabulary': ''}
usage = """usage: {}
   -json       FILE : json file containing tokenization options (mode, vocabulary, ...)
   -mode       MODE : tokenization mode: aggressive, conservative [aggressive]
   -bpe_model  FILE : 
   -joiner_annotate : use joiner annotate
   -lc              : lowercase all data (use -case_feature)
   -vocabulary FILE : vocabulary file []
   -h               : this message
""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=="-json" and len(sys.argv):
    	fjson = sys.argv.pop(0)

    elif tok=="-lc":
        tokopts['case_feature'] = True

    elif tok=="-joiner_annotate":
		tokopts['joiner_annotate'] = True

    elif tok=="-mode" and len(sys.argv):
		tokopts['mode'] = sys.argv.pop(0)

    elif tok=="-bpe_model" and len(sys.argv):
		tokopts['bpe_model_path'] = sys.argv.pop(0)

    elif tok=="-vocabulary" and len(sys.argv):
  		tokopts['vocabulary'] = sys.argv.pop(0)

    elif tok=="-h":
		sys.stderr.write("{}".format(usage))
		sys.exit()
    else:
    	sys.stderr.write('error: unparsed {} option\n'.format(tok))
    	sys.stderr.write("{}".format(usage))
    	sys.exit()

### read tokenizer options from json file
if fjson is not None:
    with open(fjson) as jsonfile: 
        tokopts = json.load(jsonfile)

tokenizer = build_tokenizer(tokopts)

for line in sys.stdin:
    line, _ = tokenizer.tokenize(str(line.strip('\n')))
    print("{}".format(" ".join(line)))
