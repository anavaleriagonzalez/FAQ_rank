import os
import re
import yaml
import xml.etree.ElementTree as ET

def get_filepath(datafolder):
    """ save the xml files except for the multiline files"""
    return [os.path.join(datafolder, f) for f in os.listdir(datafolder) if "xml" in f and "multiline" not in f]

def filepaths_fromdict(datafolders):
    """takes in the datafolders dictionary from the config file and outputs a dictionary of format:
     {set_: [datafile1, datafile2,datafile3]}"""

    all_filepaths = {}
    for key, value in datafolders.items():
        all_filepaths[key] = get_filepath(value)
    return all_filepaths

def load_configs(config_file):

    """loads config file containing the directory 
    of the different data samples"""

    with open(config_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

        datafolders = {}
        for section in cfg:
            datafolders[section]= cfg[section]
    all_filepaths = filepaths_fromdict(datafolders)

    return all_filepaths


