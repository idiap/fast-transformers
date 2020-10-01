#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Boilerplate code for dealing with fast_transformers modules."""


def make_mirror(src_module, dst_module):
    """Sets the parameters of src_module to dst_module so that they share the
    same parameters.

    Most noteable usecase is to make a recurrent transformer mirror of a batch
    transformer for fast inference.

    Arguments
    ---------
        src_module: Module to take the parameters from
        dst_module: Module to set the parameters to

    Returns
    -------
        None, it changes dst_module in place
    """
    def setattr_recursive(mod, key, value):
        key, *next_key = key.split(".", maxsplit=1)
        if not next_key:
            setattr(mod, key, value)
        else:
            setattr_recursive(getattr(mod, key), next_key[0], value)

    for name, param in src_module.named_parameters():
        setattr_recursive(dst_module, name, param)
