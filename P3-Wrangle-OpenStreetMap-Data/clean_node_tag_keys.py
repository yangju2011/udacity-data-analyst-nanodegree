# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:28:31 2016

@author: jy2487
"""

import xml.etree.cElementTree as ET

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_original.osm"  # Replace this with your osm file
NEW_OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_clean_node_tag_keys.osm"

def convert_key(letter1,letter2,key):
    """exchange letter1 with letter2 in a key, return the updated key"""
    l1 = key.find(letter1)
    better_key = key[:l1]+letter2+key[l1+1:]
    return better_key

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag
    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()
            
def clean_node_tag_keys(filename,newfilename):   
    """Create a new xml file with node tag keys and values updated"""
    with open(filename, "rb") as infile, open(newfilename, "wb") as outfile:
        outfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outfile.write('<osm>\n  ')
    
        for elem in get_element(infile):
            if elem.tag == "node": #only check node here, some node id has been duplicated
                if elem.find("tag") != -1:
                    for tag in elem.iter("tag"): 
                        if tag.attrib['k'] == 'FIXME' or tag.attrib['k'] == 'fixme' : 
                            elem.remove(tag) # remove FIXME and fixme tag
                            if tag.attrib['v'] == 'continue' or tag.attrib['v'] == 'continues': #replace fixme tag with noexit tag
                                ET.SubElement(elem,'tag', {'k':'noexit', 'v':'yes'}) #https://bugs.python.org/file2174/cElementTreeTest.py
                            elif tag.attrib['v'] == 'address & hours':
                                ET.SubElement(elem,'tag', {'k':'addr:city', 'v':'Brooklyn'})
                                ET.SubElement(elem,'tag', {'k':'addr:country', 'v':'US'})
                                ET.SubElement(elem,'tag', {'k':'addr:housenumber', 'v':'1993'})
                                ET.SubElement(elem,'tag', {'k':'addr:postcode', 'v':'11233'})
                                ET.SubElement(elem,'tag', {'k':'addr:state', 'v':'NY'})
                                ET.SubElement(elem,'tag', {'k':'addr:street', 'v':'Atlantic Ave'})
                                ET.SubElement(elem,'tag', {'k':'opening_hours', 'v':'24/7'})                        
                        elif ' ' in tag.attrib['k']: #audit_node_tags and covnert problem keys containint ' ' or '.'
                            tag.attrib['k'] = convert_key(' ','_',tag.attrib['k']) # convert space to underscore (' ','_',key) for 'Rehearsal space'
                        elif '.' in tag.attrib['k']:
                            tag.attrib['k'] = convert_key('.',':',tag.attrib['k']) # convert dot to colon ('.',':',key) for 'cityracks.housenum'
                        elif 'alt_name:' in tag.attrib['k']:
                            tag.attrib['k'] = convert_key(':','_',tag.attrib['k'])  
            outfile.write(ET.tostring(elem, encoding='utf-8'))
        outfile.write('</osm>')
                        
clean_node_tag_keys(OSM_FILE,NEW_OSM_FILE)