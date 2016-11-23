# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:10:57 2016

@author: Ju
"""

import xml.etree.cElementTree as ET
import re

OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_clean_node_tag_keys.osm"  # Replace this with your osm file
NEW_OSM_FILE = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan_cleaned.osm"

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Alley","Plaza","Commons","Broadway","Expressway","Terrace","Center","Circle",
            "Crescent","Highway","Way"]

# convert all lower case and variations to expected street names
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

# add "Street" to numbered street names   
street_number_re = re.compile(r'((1\s*st)|(2\s*nd)|(3\s*rd)|([0,4,5,6,7,8,9]\s*th))$')

direction_re = re.compile(r'(\s(S|N|W|E)$)')
direction_mapping = { " S": " South",
                     " N":" North",
                     " W":" West",
                     " E":" East"
                     } 
                       
    
def is_street_name(elem):
    """check if elem is a street name"""
    return (elem.attrib['k'] == "addr:street")

def is_postcode(elem):
    """check if elem is a postcode"""
    return (elem.attrib['k'] == "addr:postcode" or elem.attrib['k'] == "postal_code")
    
def better_name(name):
    """Clean street name to add Street to numbered name, exchange direction abbreviation and exchange street name abbreviation
    Return updated names"""
    m1 = street_number_re.search(name)
    m2 = direction_re.search(name) 
    m3 = street_type_re.search(name)
    
    if m1 and ('street' not in name and 'Street' not in name): # add Street to the end of the name 42nd --> 42nd Street
        pattern = m1.group()
        start_index = m1.start()
        length = len(pattern)
        end_index = start_index + length
        name = name[:end_index] + ' Street'    
        return name
    elif m2:
        pattern = m2.group()
        name = name[:-2] + direction_mapping[pattern] # change the very last letter to full name
        return name
    elif m3:
        pattern = m3.group()
        if pattern in street_mapping:
            start_index = m3.start()
            name = name[:start_index] + street_mapping[pattern] 
        return name
    else:
        return name

def better_postcode(postcode):
    """Clean postcode to a uniform format of 5 digit; Return updated postcode"""
    if re.findall(r'^\d{5}$', postcode): #basic format 10002
        clean_postcode = postcode
        return clean_postcode
    elif re.findall(r'(^\d{5})-\d{4}$', postcode): #long format 10002-0001
        clean_postcode = re.findall(r'(^\d{5})-\d{4}$', postcode)[0]
        return clean_postcode
    elif re.findall(r'NY\s*\d{5}', postcode): # New York NY 10065
        clean_postcode =re.findall(r'\d{5}', postcode)[0]  
        return clean_postcode  
    else:
        return None 
    
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
        
def clean_street_names(filename,newfilename):   
    """Create a new xml file with all tag keys of street names and postcode updated"""
    with open(filename, "rb") as infile, open(newfilename, "wb") as outfile:
        outfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outfile.write('<osm>\n  ')
        for elem in get_element(infile):
            if elem.tag == "node" or elem.tag == "way": #only check way and node
                if elem.find("tag") != -1:
                    for tag in elem.iter("tag"): 
                        if is_street_name(tag):
                            street_name = tag.attrib['v']
                            street_name = better_name(street_name)
                            tag.attrib['v'] = street_name           
                        elif is_postcode(tag):
                            postcode = tag.attrib['v'] 
                            if better_postcode(postcode) == None:
                                elem.remove(tag) #remove incorrect postcode 
                            else:
                                postcode = better_postcode(postcode)
                                tag.attrib['v']  = postcode                       
            outfile.write(ET.tostring(elem, encoding='utf-8')) 
        outfile.write('</osm>')
                        
clean_street_names(OSM_FILE,NEW_OSM_FILE)