import csv
import yaml
import new_code.kg_exploration as KE
import new_code.kg_construction as KC
import new_code.kg_filtering as KF

def kg_exploration(universal_config):
    kgexplorer = KE.KGExploration(**universal_config)
    file_content_list = kgexplorer.read_seed_files()

    chunk_list, entity_list = kgexplorer.process_entity_extraction(file_content_list)
    kgexplorer.save_extracted_entities(chunk_list, entity_list)

    chunk_list, entity_pair_list = kgexplorer.read_entity_infos()
    relation_triple_list = kgexplorer.process_relation_extraction(chunk_list, entity_pair_list)
    kgexplorer.save_extracted_relations(chunk_list, entity_pair_list, relation_triple_list)

    chunk_list, entity_list = kgexplorer.read_entity_infos()
    entity_type_list = kgexplorer.process_entity_type_labeling(chunk_list, entity_list)
    kgexplorer.save_labeled_entity_types(chunk_list, entity_list, entity_type_list)

    unique_entity_type_list = kgexplorer.read_entity_types()
    entity_type_definitions, entity_types = kgexplorer.entity_type_fusion(unique_entity_type_list)
    kgexplorer.save_fused_entity_types(entity_types, entity_type_definitions)

    unique_relation_type_list = kgexplorer.read_relation_types()
    relation_types, relation_type_definitions = kgexplorer.relation_type_fusion(unique_relation_type_list)
    kgexplorer.save_fused_relation_types(relation_types, relation_type_definitions)

def kg_construction(universal_config):
    kgconstructor = KC.KGConstruction(**universal_config)
    all_file_list = kgconstructor.read_all_files()
    entity_type_str = kgconstructor.get_entity_type()
    relation_type_str = kgconstructor.get_relation_type()
    kgconstructor.process_entity_extraction(all_file_list, entity_type_str, kgconstructor.entity_file_path)
    text_list = []
    entity_list = []
    entity_pairs = []
    with open(kgconstructor.entity_file_path,encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            text_list.append(row[0])
            entity_list.append(row[1])
            entity_pairs.append(row[2])
    kgconstructor.process_relation_extraction(text_list, relation_type_str, entity_list, entity_pairs, kgconstructor.relation_file_path)

def kg_filtering(universal_config):
    kgrefiner = KF.KGFiltering(**universal_config)
    type_triples = kgrefiner.generate_initial_schema(kgrefiner.entity_type_path, kgrefiner.relation_type_path, kgrefiner.initial_schema_path)
    type_triples_in_kg = kgrefiner.get_data(kgrefiner.initial_kg_path)
    valid_type_triple = kgrefiner.generate_refined_schema(type_triples, type_triples_in_kg, kgrefiner.refine_schema_path)
    kgrefiner.generate_refined_kg(kgrefiner.initial_kg_path, kgrefiner.refine_kg_path, valid_type_triple)

def main():
    with open('../config.yaml', 'r') as config_file:
        universal_config = yaml.safe_load(config_file)
    kg_exploration(universal_config)
    kg_construction(universal_config)
    kg_filtering(universal_config)

if __name__ == "__main__":
    main()