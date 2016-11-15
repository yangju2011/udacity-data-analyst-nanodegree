# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:44:39 2016

@author: Ju
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Alley","Plaza","Commons","Broadway","Expressway","Terrace","Center","Circle",
            "Crescent","Highway","Way"]


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group() 
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
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

# 1.5.2 convert all lower case and variations to expected street names
street_mapping = { 
                  "Ave":"Avenue",
                  "Ave.":"Avenue",
                  "Avene":"Avenue",
                  "Aveneu":"Avenue",
                  "ave":"Avenue",
                  "avenue":"Avenue",
                  "Blv.":"Boulevard",
                  "Blvd":"Boulevard",
                  "blvd":"Boulevard",
                  "Broadway.":"Broadway",
                  "Ctr":"Center",
                  "Pkwy":"Parkway",
                  "Plz":"Plaza",
                  "Rd":"Road",
                  "ST":"Street",
                  "St":"Street",
                  "St.":"Street",
                  "Steet":"Street",
                  "Streeet":"Street",
                  "st":"Street",
                  "street":"Street"
                  }

def update_name(name, mapping):
    for k in mapping:
        if name.find(k) !=-1: 
            start = name.find(k)
            name = name[:start]+mapping[k]
    return name

for st_type, ways in st_types.iteritems():
   for name in ways:
      better_name = update_name(name, street_mapping)
      print better_name
      
# 1.5.3 add "Street" to numbered street names   
street_number_re = re.compile(r'((1\s*st)|(2\s*nd)|(3\s*rd)|([0,4,5,6,7,8,9]\s*th))$')

def update_number_street(name,search_target):
    m = search_target.search(name)
    if m and ('street' not in name and 'Street' not in name):
        pattern =m.group()
        start_index = m.start()
        length = len(pattern)
        end_index = start_index + length
        name = name[:end_index] + ' Street'
        return name

for st_type, ways in st_types.iteritems():
   for name in ways:
      better_number_name = update_number_street(name,street_number_re)

# 1.5.4 convert to full name direction 
direction_re = re.compile(r'(\s+(S|N|W|E)\s*$)')
direction_mapping = { "S": "South",
                     "N":"North",
                     "W":"West",
                     "E":"East"
                     }

def full_direction(name,search_target,mapping):
    m = search_target.search(name)
    if m:
        name = update_name(name, mapping)
    return name
        
for st_type, ways in st_types.iteritems():
   for name in ways:
      better_direction_name = full_direction(name,direction_re,direction_mapping)
      
      
      
      
      
      
      
      
      
      
      
      
      
      