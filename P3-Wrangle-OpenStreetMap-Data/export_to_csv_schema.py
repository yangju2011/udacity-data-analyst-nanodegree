# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:59:22 2016

@author: Ju
"""

import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import myschema

SCHEMA = myschema.schema

OSM_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/manhattan.osm"
SAMPLE_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/sample.osm"

NODES_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/nodes.csv"
NODE_TAGS_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/nodes_tags.csv"
WAYS_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/ways.csv"
WAY_NODES_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/ways_nodes.csv"
WAY_TAGS_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+',re.IGNORECASE)
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]', re.IGNORECASE)
# here, problem chars are discarded

SCHEMA_PATH = "D:/Dropbox/Data-Analysis/p3/OSM/schema.py"

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  
    
    if element.tag == 'node':
        for key in element.attrib:
            if key in NODE_FIELDS:
                node_attribs[key] = element.attrib[key] 
        if element.find("tag") != -1:
            for tag in element.iter("tag"): 
                if PROBLEMCHARS.search(tag.attrib['k']) == None: 
                    node_tag = {}                    
                    if LOWER_COLON.search(tag.attrib['k']) != None:
                        colon_i = tag.attrib['k'].find(':') #only find the first :, ignore additional colon
                        tag_type = tag.attrib['k'][:colon_i]
                        key = tag.attrib['k'][colon_i+1:]
                    else:
                        tag_type = default_tag_type
                        key = tag.attrib['k']
                    
                    node_tag['id'] = element.attrib['id']
                    node_tag['key'] = key
                    node_tag['value'] = tag.attrib['v']
                    node_tag['type'] = tag_type
                    
                    tags.append(node_tag)
                    
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        for key in element.attrib:
            if key in WAY_FIELDS:
                way_attribs[key] = element.attrib[key] 
                
        if element.find("tag") != -1:
            for tag in element.iter("tag"): 
                #return <tag k = "a:n:v", v = >
                if PROBLEMCHARS.search(tag.attrib['k']) == None: 
                    way_tag = {}
                    #not match, then it works, otherwise, not consider this tag
                    
                    if LOWER_COLON.search(tag.attrib['k']) != None:
                        colon_i = tag.attrib['k'].find(':') #only find the first :, ignore additional colon
                        tag_type = tag.attrib['k'][:colon_i]
                        key = tag.attrib['k'][colon_i+1:]
                    else:
                        tag_type = default_tag_type
                        key = tag.attrib['k']
                    
                    way_tag['id'] = element.attrib['id']
                    way_tag['key'] = key
                    way_tag['value'] = tag.attrib['v']
                    way_tag['type'] = tag_type
                    
                    tags.append(way_tag)
        if element.find("nd") != -1: 
            #find way_nodes
            i = 0
            for node in element.iter('nd'):
                way_node = {}
                way_node['id']= element.attrib['id']
                way_node['node_id'] = node.attrib['ref']
                way_node['position'] = i
                way_nodes.append(way_node)
                i = i + 1
        
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    # process_map(SAMPLE_PATH, validate=True)
    process_map(OSM_PATH, validate=False)
