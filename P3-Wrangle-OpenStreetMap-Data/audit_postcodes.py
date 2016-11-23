# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:30:48 2016

@author: Ju
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:44:39 2016

@author: Ju
"""

import xml.etree.cElementTree as ET
import re
import pprint

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_original.osm"
postcode_re = re.compile(r'^\d{5}([\-]?\d{4})?$')
# http://stackoverflow.com/questions/578406/what-is-the-ultimate-postal-code-and-zip-regex
# use 5 digit or 10 digit postcode as expected format

def audit_postcode(postcodes,value):
    """Audit the format of value and return unexpecteed value"""
    m = postcode_re.search(value)
    if not m: #if the value is not a postcode
        postcodes.add(value)

def is_postcode(elem):
    """Check if elem is postcode"""
    return (elem.attrib['k'] == "addr:postcode" or elem.attrib['k'] == "postal_code")

def audit(osmfile):
    """Return expected postcode values"""
    osm_file = open(osmfile, "r")
    postcodes = set()
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way": #only check way and node
            for tag in elem.iter("tag"): 
                if is_postcode(tag):
                    audit_postcode(postcodes,tag.attrib['v'])
    osm_file.close()
    return postcodes
      
postcodes = audit(OSM_FILE)
print "unexpected postcode values:"
pprint.pprint(postcodes)
    
    