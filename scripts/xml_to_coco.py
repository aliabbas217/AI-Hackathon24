from ..utils.xml2coco import main as coco_main
xml_folder = "./data/xmls"
actions_folder = "./data/groundTruth"
output_file = "./data/merged_final.json"

coco_main(xml_folder, actions_folder, output_file)