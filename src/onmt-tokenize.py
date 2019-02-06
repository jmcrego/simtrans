# -*- coding: utf-8 -*-

import sys
import json
import yaml
import os
import six
from tokenizer import build_tokenizer


fyaml = None
fjson = None
tokopts = {'mode': 'aggressive', 'vocabulary': ''}
usage = """usage: {}
   -json       FILE : json file containing tokenization options (mode, vocabulary, bpe_model_path, joiner_annotate, case_feature, ...)
   -yaml       FILE : yaml file containing tokenization options (mode, vocabulary, bpe_model_path, joiner_annotate, case_feature, ...)
   -mode       MODE : tokenization mode: aggressive, conservative [aggressive]
   -bpe        FILE : bpe codes to apply subtokenization (use -bpe_model_path)
   -joiner          : use joiner annotate (use -joiner_annotate))
   -lc              : lowercase all data (use -case_feature)
   -vocabulary FILE : vocabulary file
   -h               : this message

+ yaml file is not used if json file is passed
+ json/yaml options override command-line options
+ See https://github.com/OpenNMT/Tokenizer/blob/master/bindings/python/README.md for the list of available options of pyonmttok.
""".format(sys.argv.pop(0))

while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=="-json" and len(sys.argv):
    	fjson = sys.argv.pop(0)

    elif tok=="-yaml" and len(sys.argv):
    	fyaml = sys.argv.pop(0)
        
    elif tok=="-lc":
        tokopts['case_feature'] = True

    elif tok=="-joiner":
	tokopts['joiner_annotate'] = True

    elif tok=="-mode" and len(sys.argv):
	tokopts['mode'] = sys.argv.pop(0)

    elif tok=="-bpe" and len(sys.argv):
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

if fjson is not None: ### read tokenizer options from json file
    with open(fjson) as jsonfile: 
        tokopts = json.load(jsonfile)
elif fyaml is not None: ### read tokenizer options from yaml file
    with open(fyaml) as yamlfile:
        tokopts = yaml.load(yamlfile)

t = build_tokenizer(tokopts)
for line in sys.stdin:
    line, _ = t.tokenize(str(line.strip('\n')))
    print("{}".format(" ".join(line)))
