# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:21:11 2016

@author: Ju
"""

import xml.etree.cElementTree as ET
import re
import pprint

def audit_tags(osmfile):
    """Audit an osmfile and return sets of node tag keys, node tag values, way tag keys and way tag values"""
    osm_file = open(osmfile, "r")
    node_tags_keys = set()
    node_tags_values = set()
    way_tags_keys = set()
    way_tags_values = set()

    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node": #only check way and node
            for tag in elem.iter('tag'):
                node_tags_keys.add(tag.attrib['k'])
                node_tags_values.add(tag.attrib['v'])
        if elem.tag == "way": #only check way and node
            for tag in elem.iter('tag'):
                way_tags_keys.add(tag.attrib['k'])
                way_tags_values.add(tag.attrib['v'])
    osm_file.close()
    return node_tags_keys, node_tags_values, way_tags_keys, way_tags_values 

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_original.osm"  # Replace this with your osm file
node_tags_keys, node_tags_values, way_tags_keys, way_tags_values  = audit_tags(OSM_FILE)

lower = re.compile(r'^([a-z]|_)*$', re.IGNORECASE)
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$', re.IGNORECASE)
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]', re.IGNORECASE)
# case insensitive search

def tag_keys(tag_keys_set):
    """Return sets of different types of keys"""
    lower_keys = set()
    lower_colon_keys = set()
    problemchars_keys = set()
    others_keys = set()
    
    for key in tag_keys_set:
        if lower.search(key):
            lower_keys.add(key)
        elif lower_colon.search(key):
            lower_colon_keys.add(key)
        elif problemchars.search(key):
            problemchars_keys.add(key)
        else:
            others_keys.add(key)
            
    return lower_keys,lower_colon_keys,problemchars_keys,others_keys

node_lower_keys,node_lower_colon_keys,node_problemchars_keys,node_others_keys = tag_keys(node_tags_keys)

node_key_types = {"total": len(node_tags_keys),"lower":len(node_lower_keys),"lower colon":len(node_lower_colon_keys),"problemchars":len(node_problemchars_keys),"others":len(node_others_keys)}

print "The number of keys for each type is: "
pprint.pprint(node_key_types)

'''uncomment the print lines below for tag keys of each category'''

#print "node tag keys containg lower case letters" 
#pprint.pprint(node_lower_keys)

#print "node tag keys containing 1 colon" 
#pprint.pprint(node_lower_colon_keys)

#print "node tag keys with problem chars" 
pprint.pprint(node_problemchars_keys)

#print "other node tag keys" 
#pprint.pprint(node_others_keys)