# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:56:04 2016

@author: Ju
"""
import csv

def add_unique_id(filename, newfilename,idname):
    with open(filename, "rb") as infile, open(newfilename, "wb") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow([idname] + next(reader))
        writer.writerows([i] + row for i, row in enumerate(reader, 1))
#http://stackoverflow.com/questions/23261852/adding-column-in-csv-python-and-enumerating-it

add_unique_id("nodes_tags.csv", "nodes_tags_id.csv","NodetagId")
add_unique_id("ways_nodes.csv", "ways_nodes_id.csv","WaynodeId")
add_unique_id("ways_tags.csv", "ways_tags_id.csv","WaytagId")

def skip_header(filename,newfilename):
    with open(filename, "rb") as infile, open(newfilename, "wb") as outfile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row)

skip_header("nodes.csv","nodes_noheader.csv")
skip_header("ways.csv","ways_noheader.csv")
skip_header("nodes_tags_id.csv","nodes_tags_noheader.csv")
skip_header("ways_nodes_id.csv","ways_nodes_noheader.csv")
skip_header("ways_tags_id.csv","ways_tags_noheader.csv")
