# -*- coding: utf-8 -*-

import sys
import os
import six
import pyonmttok

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    local_args = {}
    for k, v in six.iteritems(args):
        if isinstance(v, six.string_types):
            local_args[k] = v.encode('utf-8')
        else:
            local_args[k] = v

    if not 'mode' in local_args:
        sys.stderr.write('error: missing mode in tokenizer options\n')
        sys.exit()
        
    sys.stderr.write("tokenizer args = {}\n".format(local_args))
    mode = local_args['mode']
    del local_args['mode']
    if 'vocabulary' in local_args: del local_args['vocabulary']

    return pyonmttok.Tokenizer(mode, **local_args)
