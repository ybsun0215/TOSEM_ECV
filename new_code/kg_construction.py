import json
from itertools import combinations
import os
import ast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from tqdm import tqdm
from util import load_message, call_llm

class KGConstruction:
    def __init__(self, **constructor_config) -> None:
        # API_key
        self.key_list = constructor_config["API_key_list"]
        # Model name
        self.model_name = constructor_config["llm_name"]
        # all_file_path
        self.all_file_path = constructor_config["all_file_path"]
        # prompt_path
        self.entity_extraction_prompt_path = constructor_config["kc_entity_extraction_prompt"]
        self.entity_extraction_example_path = constructor_config["kc_entity_extraction_examples"]
        self.relation_extraction_prompt_path = constructor_config["kc_relation_extraction_prompt"]
        self.relation_extraction_example_path = constructor_config["kc_relation_extraction_examples"]
        # type_path
        self.entity_type_path = constructor_config["save_ke_entity_type_path"]
        self.relation_type_path = constructor_config["save_ke_relation_type_path"]
        # save path
        self.entity_file_path = constructor_config["save_kc_entity_path"]
        self.relation_file_path = constructor_config["save_kc_relation_path"]

    def read_all_files(self):
        # check seed_path
        if os.path.isdir(self.all_file_path):
            files = os.listdir(self.all_file_path)
            files_list = [os.path.join(self.all_file_path, file) for file in files]
        else:
            files_list = []
            print("Invalid seed path")
        # read file_content
        file_content_list = []
        for path in files_list:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    result = list(reader)
                    for i in range(len(result)):
                        file_content_list.append(result[i][0])
            except Exception as e:
                print(e)
        return file_content_list

    def get_entity_type(self):
        entity_type_list = []
        with open(self.entity_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                entity_type_list.append({row[0]: row[1]})

        result_dict = {}
        for item in entity_type_list:
            result_dict.update(item)
        entity_type_str = '\n'.join([f"'{k}': '{v}'" for k, v in result_dict.items()])
        return entity_type_str

    def entity_extraction(self, chunk, entity_type_str, api_key, index):
        try:
            prompt_content = open(self.entity_extraction_prompt_path, 'r', encoding='utf-8').read()
            llm_input = load_message(prompt_content)

            llm_input.append({
                "role": "assistant",
                "content": "Please provide the text and all the entity type definitions, and I will extract the API entities from the text according to your requirements."
            })

            with open(self.entity_extraction_example_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)

            for example in examples:
                entity_types_text = "entity types and their definitions: \n"
                for entity_type, definition in example['entity_types'].items():
                    entity_types_text += f"{entity_type}: {definition}\n"

                llm_input.append({
                    "role": "user",
                    "content": f"text: {example['text']}\n{entity_types_text}"
                })

                llm_input.append({
                    "role": "assistant",
                    "content": example['output']
                })

            llm_input.append({
                "role": "user",
                "content": f"text: {chunk}\nentity types and their definitions: {entity_type_str}"
            })

            entity_result = call_llm(llm_input, api_key, self.model_name)

            if ":" in entity_result:
                entities = entity_result.split("\n")[1]
            else:
                entities = ""
        except Exception as e:
            print(f"Error in entity_extraction for index {index}: {str(e)}")
            entities = ""

        return index, entities

    def save_extracted_entities(self, chunk_list, entity_list, entity_pair_list):
        result = []
        for idx in range(len(chunk_list)):
            task = {
                "chunk": chunk_list[idx],
                "entities": entity_list[idx],
                "entity_pair": entity_pair_list[idx],
            }
            result.append(task)
        return result

    def process_entity_extraction(self, chunk_list, entity_type_str, save_path):
        all_result = []
        entity_list = [None] * len(chunk_list)
        entity_pair_list = [None] * len(chunk_list)
        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(self.entity_extraction, chunk, entity_type_str,
                                self.key_list[idx % len(self.key_list)], idx): idx for idx, chunk in
                enumerate(chunk_list)
            }
            for future in tqdm(as_completed(futures), desc="Entity Extraction", total=len(futures)):
                try:
                    index, entities = future.result()
                    entity_list[index] = entities
                    entity_split_list = [item.split(": ")[0].strip() for item in entities.split("; ")]
                    entity_pair_list[index] = list(combinations(entity_split_list, 2))
                    result = self.save_extracted_entities(
                        [chunk_list[index]], [entity_list[index]], [entity_pair_list[index]])
                    all_result.extend(result)
                except Exception as e:
                    print(f"Error processing chunk at index {index}: {e}")
        df = pd.DataFrame(all_result)
        all_columns = df.columns.tolist()
        df = df[all_columns[:3]]
        df.to_csv(save_path, index=False, header=False)

    def get_relation_type(self):
        relation_type_list = []
        with open(self.relation_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                relation_type_list.append({row[0]: row[1]})
        result_dict = {}
        for item in relation_type_list:
            result_dict.update(item)
        relation_type_str = '\n'.join([f"'{k}': '{v}'" for k, v in result_dict.items()])
        return relation_type_str

    def relation_extraction(self, chunk, relation_type_str, entity_pair, api_key, index):
        try:
            if isinstance(entity_pair, str):
                entity_pair = ast.literal_eval(entity_pair)

            if not entity_pair:
                return index, []

            valid_pairs = [p for p in entity_pair if isinstance(p, (list, tuple)) and len(p) == 2]
            if not valid_pairs:
                return index, []

            entity_pair_str = "; ".join(f"({a}, {b})" for a, b in valid_pairs)

            with open(self.relation_extraction_prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            llm_input = load_message(prompt_content)

            llm_input.append({
                "role": "assistant",
                "content": "Please provide the text, entity pairs, and all the relationship types definitions, and I will identify the relevant relationship triples from the text based on your requirements."
            })

            with open(self.relation_extraction_example_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)

            for example in examples:
                relation_types_text = "relation types and their definitions: \n"
                for relation_type, definition in example['relation_types'].items():
                    relation_types_text += f"{relation_type}: {definition}\n"

                llm_input.append({
                    "role": "user",
                    "content": f"text: {example['text']}\nentity pairs: {example['entity_pairs']}\n{relation_types_text}"
                })

                llm_input.append({
                    "role": "assistant",
                    "content": example['output']
                })

            llm_input.append({
                "role": "user",
                "content": f"text: {chunk}\nentity pairs: {entity_pair_str}\nrelation types and their definitions: {relation_type_str.lower()}"
            })

            # 调用LLM
            relation_result = call_llm(llm_input, api_key, self.model_name)

            # 解析结果
            relations = relation_result.splitlines()[1:] if relation_result else []
            return index, relations

        except Exception as e:
            print(f"❌ relation_extraction failed for index {index}")
            print("entity_pair(raw):", entity_pair)
            print("error:", repr(e))
            return index, []

    def convert2_relation_triples(self, entity_list, relation_list):
        final_output_list = []
        for entities_str, relations in zip(entity_list, relation_list):
            entity_type_mapping = {}
            entities = entities_str.split('; ')
            for entity in entities:
                if entity:
                    entity_name, entity_type = entity.split(': ')
                    entity_type_mapping[entity_name] = entity_type
            output_per_relation = []
            for relation in relations:
                relation_type, entity_triplet = relation.split(': ')
                if entity_triplet.startswith('(') and entity_triplet.endswith(')'):
                    new_entity_triple = entity_triplet[1:-1]
                else:
                    new_entity_triple = entity_triplet
                entity1, relation_name, entity2 = new_entity_triple.split(', ')
                type1 = entity_type_mapping.get(entity1, "unknown").replace(";","")
                type2 = entity_type_mapping.get(entity2, "unknown").replace(";","")
                type_triple = f"({type1}, {relation_type}, {type2})"
                instance_triple = f"({entity1}, {relation_name}, {entity2})"
                output_per_relation.append(f"{type_triple}: {instance_triple}")
            final_output_list.append(output_per_relation)
        return final_output_list

    def save_extracted_relations(self, chunk_list, entity_list, relation_list):
        result = []
        triple_with_type = self.convert2_relation_triples(entity_list, relation_list)
        for idx in range(len(chunk_list)):
            task = {
                "chunk": chunk_list[idx],
                "relations": relation_list[idx],
                "relation_with_type": triple_with_type[idx],
            }
            result.append(task)
        return result

    def process_relation_extraction(self, chunk_list, relation_type_str, entity_list, entity_pair_list, save_path):
        all_result = []
        relation_list = [None] * len(chunk_list)
        with ThreadPoolExecutor(max_workers=len(self.key_list)) as executor:
            futures = {
                executor.submit(
                    self.relation_extraction, chunk, relation_type_str, entity_pair,
                    self.key_list[idx % len(self.key_list)], idx): idx for idx, (chunk, entity_pair) in
                enumerate(zip(chunk_list, entity_pair_list))
            }
            for future in tqdm(as_completed(futures), desc="Relation Extraction", total=len(futures)):
                try:
                    index, relation_triples = future.result()
                    relation_list[index] = relation_triples
                    result = self.save_extracted_relations(
                        [chunk_list[index]], [entity_list[index]], [relation_list[index]])
                    all_result.extend(result)
                except:
                    all_result.extend([{"chunk": chunk_list[index],"relations":"","relation_with_type":"[]"}])
        df = pd.DataFrame(all_result)
        df.to_csv(save_path, index=False, header=False)
        return relation_list
