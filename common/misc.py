# -*- coding: utf-8 -*-

def mergeTwoDicts(dest, src, copy=True):
    '''
    Merge two dicts by copying/updating all keys from src dict to dest dict.

    Parameters
    ----------
    dest: dict
        Dict object to be updated
    src: dict
        Dict object that updates dest.
    copy: bool
        If value is True, a copy of dest dict is made before updating.

    Returns
    -------
    merged: dict
        Dict x updated by keys in y.
    '''
    if copy:
        merged = dest.copy()
    else:
        merged = dest
    merged.update(src)
    return merged


def mergeTwoDictsDeep(dest, src, copy=True):
    '''
    Merge two dicts by recursively copying/updating all keys from src dict to
    dest dict (going into child dicts).

    Parameters
    ----------
    dest: dict
        Dict object to be updated
    src: dict
        Dict object that updates dest.
    copy: bool
        If value is True, a copy of dest dict is made before updating.

    Returns
    -------
    merged: dict
        Dict x updated by keys in y.
    '''
    if copy:
        merged = dest.copy()
    else:
        merged = dest

    for key in src:
        if key in dest:
            if isinstance(dest[key], dict):
                mergeTwoDictsDeep(merged[key], src[key], copy=False)
            else:
                merged.update(src)

    return merged
