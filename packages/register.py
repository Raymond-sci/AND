#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-25 15:08:10
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import sys

class REGISTER:
    """Singleton Register Class
    
    This class is used to register all modules in the whole package
    
    Variables:
        PACKAGES_2_CLASSES {dict} -- record the registered modules
    """

    PACKAGES_2_CLASSES = dict()

    @staticmethod
    def set_package(package):
        """set package
        
        This function is called when a new package has been created
        
        Arguments:
            package {str} -- package name
        """
        if not package in REGISTER.PACKAGES_2_CLASSES:
            REGISTER.PACKAGES_2_CLASSES[package] = dict()

    @staticmethod
    def get_packages():
        """get packages list
        
        Get the packages list stored in REGISTER.PACKAGES_2_CLASSES
        
        Returns:
            list -- packages list
        """
        return REGISTER.PACKAGES_2_CLASSES.keys()

    @staticmethod
    def get_package_name(module):
        """get the package name of a module
        
        Get the package name of the target module by removing the text after
        the last dot occured in the provided name
        
        Arguments:
            module {str} -- target module name
        """
        return '.'.join(module.split('.')[:-1])

    @staticmethod
    def is_package_registered(package):
        """Check whether a package has been registered
        
        Check whether a package has been registered according to whether its 
        name is occured in PACKAGES_2_CLASSES
        
        Arguments:
            package {str} -- target package name
        
        Returns:
            bool -- indicating whether package has been registered
        """
        return package in REGISTER.PACKAGES_2_CLASSES

    @staticmethod
    def _check_package(package):
        assert package in REGISTER.PACKAGES_2_CLASSES, ('No package named [%s] '
                                                'has been registered' % package)

    @staticmethod
    def get_classes(package):
        """get all classes on the target package
        
        Get all registered classes on the target package according to the
        PACKAGES_2_CLASSES dictory
        
        Arguments:
            package {str} -- package name used when registering
        
        Returns:
            dict -- {'name1' : class1, 'name2' : class2, ...}
        """

        # make sure the packages has been registered
        REGISTER._check_package(package)

        return REGISTER.PACKAGES_2_CLASSES[package]

    @staticmethod
    def is_class_registered(package, name):
        return (package in REGISTER.PACKAGES_2_CLASSES and
            name in REGISTER.PACKAGES_2_CLASSES[package])

    @staticmethod
    def get_class(package, name):
        """get class
        
        This function is used to get the class named `name` from
        package `package`
        
        Arguments:
            package {str} -- package name, should be the same as the one when registering
            name {str} -- class name, should be the same as the one when registering
        
        Returns:
            [Object] -- found class
        """

        # make sure the package has been registered
        REGISTER._check_package(package)

        pack = REGISTER.PACKAGES_2_CLASSES[package]

        assert name in pack, ('No class named [%s] has been registered '
                              'in package [%s]' % (name, package))

        return pack[name]

    @staticmethod
    def set_class(package, name, cls):
        """set class
        
        This function is used to register a class with a specific name in
        the target package
        
        Arguments:
            package {str} -- target package name
            name {str} -- class name which will be used when query
            cls {Object} -- class object
        """
        
        # get the corresponding package name
        # package = '.'.join(package.split('.')[:-1])
        if not package in REGISTER.PACKAGES_2_CLASSES:
            REGISTER.set_package(package)

        # if the `cls` arg is a string, then get the real class object
        if isinstance(cls, str):
            cls = sys.modules[cls]

        REGISTER.PACKAGES_2_CLASSES[package][name] = cls