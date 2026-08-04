import csv
import ast

class KGFiltering:
    def __init__(self, **refiner_config) -> None:
        self.entity_type_path = refiner_config["save_ke_entity_type_path"]
        self.relation_type_path = refiner_config["save_ke_relation_type_path"]
        self.initial_schema_path = refiner_config["save_kr_initial_schema_path"]
        self.initial_kg_path = refiner_config["save_kc_relation_path"]
        self.refine_schema_path = refiner_config["save_kr_refine_schema_path"]
        self.refine_kg_path = refiner_config["save_kr_refine_kg_pextractionath"]

        self.predefined_support = refiner_config["predefined_support"]
        self.predefined_confidence = refiner_config["predefined_confidence"]
        self.predefined_lift = refiner_config["predefined_lift"]
    def generate_initial_schema(self, entity_type_path, relation_type_path, save_path):
        print("************************* KG Refinement *************************")
        entity_types = []
        with open(entity_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                entity_types.append(row[0].strip("'"))
        relation_types = []
        with open(relation_type_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                relation_types.append(row[0].strip("'"))
        type_triplets = []
        for rel in relation_types:
            for ent1 in entity_types:
                for ent2 in entity_types:
                    type_triplets.append((ent1.lower(), rel.lower(), ent2.lower()))

        with open(save_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for triplet in type_triplets:
                writer.writerow([str(triplet).lower()])
        return type_triplets

    def get_data(self, initial_kg_path):
        triplets = []
        with open(initial_kg_path, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                second_column_data = ast.literal_eval(row[2])
                for item in second_column_data:
                    triplet_str = item.split(": ")[0].strip("()")
                    triplet = tuple(triplet_str.lower().split(", "))
                    triplets.append(triplet)
        return triplets

    def generate_refined_schema(self,type_triples, type_triples_in_kg, refine_schema_path):
        valid_type_triple = []
        for i in range(len(type_triples)):
            target_type_triple = type_triples[i]
            support = cal_support(target_type_triple, type_triples_in_kg)
            confidence = cal_confidence(target_type_triple, type_triples_in_kg)
            lift = cal_lift(target_type_triple, confidence, type_triples_in_kg)
            print(f"type_triple {type_triples[i]}: support {support}, confidence {confidence}, lift {lift}")
            if support >= float(self.predefined_support) and confidence >= float(self.predefined_confidence) and lift >= float(self.predefined_lift):
                if target_type_triple not in valid_type_triple:
                    valid_type_triple.append(target_type_triple)
        with open(refine_schema_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for triplet in valid_type_triple:
                writer.writerow([str(triplet)])
        return valid_type_triple

    def generate_refined_kg(self, initial_kg_path, refine_kg_path, valid_type_triple):
        with open(initial_kg_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        with open(refine_kg_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                first_column = row[0]
                second_column_data = ast.literal_eval(row[2])
                filtered_triplets = []
                for item in second_column_data:
                    triplet_str = item.split(": ")[0].strip("()")
                    triplet = tuple(triplet_str.lower().split(", "))  # 🔧 加上 .lower()
                    if triplet in valid_type_triple:
                        filtered_triplets.append(item)
                if filtered_triplets:
                    writer.writerow([first_column, str(filtered_triplets)])
                else:
                    writer.writerow([first_column, '[]'])

def cal_support(target_type_triple, type_triples_in_kg):
    n_all = len(type_triples_in_kg)
    n_a_b_r = type_triples_in_kg.count(target_type_triple)
    if n_all == 0:
        return 0
    return n_a_b_r / n_all

def cal_confidence(target_type_triple, type_triples_in_kg):
    n_a_b_r = type_triples_in_kg.count(target_type_triple)
    n_a_b = len([tup for tup in type_triples_in_kg if tup[0] == target_type_triple[0] and tup[2] == target_type_triple[2]])
    if n_a_b == 0:
        return 0
    return n_a_b_r / n_a_b

def cal_lift(target_type_triple, confidence, type_triples_in_kg):
    n_all = len(type_triples_in_kg)
    n_r = len([tup for tup in type_triples_in_kg if tup[1] == target_type_triple[1]])
    if n_r == 0 or n_all == 0:
        return 0
    return confidence / (n_r / n_all)


