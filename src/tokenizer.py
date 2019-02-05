"""Tokenization utilities."""

import os
import sys
import six

def build_tokenizer(args):
    """Builds a tokenizer based on user arguments."""
    sys.stderr.write('tokenizer args = {}\n'.format(args))
    import pyonmttok
    local_args = {}
    for k, v in six.iteritems(args):
        print(k,v)
        if isinstance(v, six.string_types):
            local_args[k] = v.encode('utf-8')
        else:
            local_args[k] = v
    mode = local_args['mode']
    del local_args['mode']
    del local_args['vocabulary']
    return pyonmttok.Tokenizer(mode, **local_args)
