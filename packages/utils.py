#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-28 12:20:22
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

def tuple_or_list(target):
    """Check is a string object contains a tuple or list
    
    If a string likes '[a, b, c]' or '(a, b, c)', then return true,
    otherwise false.
    
    Arguments:
        target {str} -- target string
    
    Returns:
        bool -- result
    """

    # if the target is a tuple or list originally, then return directly
    if isinstance(target, tuple) or isinstance(target, list):
        return target

    try:
        target = eval(target)
        if isinstance(target, tuple) or isinstance(target, list):
            return target
    except:
        pass
    return None

def get_valid_size(target):
    """get valid size
    
    if target is a tuple/list or a string of them, then convert and return
    if target is a int/float then return
    else return None
    
    Arguments:
        target {Number} -- size
    """
    ret = tuple_or_list(target)
    if ret is not None:
        return ret

    try:
        ret = int(target)
        return ret
    except:
        pass

    try:
        ret = float(target)
        return ret
    except:
        pass

    return None

