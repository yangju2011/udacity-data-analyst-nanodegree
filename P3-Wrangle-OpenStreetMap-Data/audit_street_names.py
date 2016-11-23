# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:44:39 2016

@author: Ju
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_original.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Alley","Plaza","Commons","Broadway","Expressway","Terrace","Center","Circle",
            "Crescent","Highway","Way"]


def audit_street_type(street_types, street_name):
    """Audit the type of a street name and add to a street_type set"""
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group() 
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):
    """Check if elem is a street name"""
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
    """Return a dict of street_types and street_name from an osmfile"""
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way": #only check way and node
            for tag in elem.iter("tag"): 
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

st_types = audit(OSM_FILE)

pprint.pprint(dict(st_types)) 
      
      
      
      
      
      
      
      
      
      