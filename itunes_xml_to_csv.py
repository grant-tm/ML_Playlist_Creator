import sys
import pandas as pd
import xml.etree.ElementTree as ET

def main():
    if len(sys.argv) < 2:
        return
    filename = sys.argv[1]
    xml = parse_xml(filename)
    library = extract_metadata(xml)
    pd.DataFrame(library).to_csv(filename[:-4] + '.csv', index=False)
    return

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def extract_metadata(xml_file):
    for node in xml_file:
        if node.tag == 'dict':
            xml_info = node
    for node in xml_info:
        if node.tag == 'dict':
            metadata = extract_track_metadata(node)
            return metadata
    
def extract_track_metadata(track_list):
    library = []
    for track in track_list:
        if track.tag == "dict":
            track_metadata = aggregate_metadata_fields(track)
            library.append(track_metadata)
    return library

def aggregate_metadata_fields(track):
    metadata_dictionary = {}
    field = ''
    for entry in track:
        if entry.tag == "key":
            field = entry.text
        else:
            metadata_dictionary[field] = entry.text
    return metadata_dictionary

if __name__ == "__main__":
    main()