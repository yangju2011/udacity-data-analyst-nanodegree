# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:25:26 2016

@author: Ju
"""

import xml.etree.cElementTree as ET

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan.osm"

def audit_prob_attrib(osmfile,key):
    osm_file = open(osmfile, "r")
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node": 
            for tag in elem.iter('tag'):
                if tag.attrib['k'] == key:
                        print (elem.attrib['id'],elem.attrib['uid'],tag.attrib)
    osm_file.close()
    
print "node id, user id, tags containing 'FIXME':"
audit_prob_attrib(OSM_FILE,'FIXME')
print "node id, user id, tags containing 'fixme':"
audit_prob_attrib(OSM_FILE,'fixme')