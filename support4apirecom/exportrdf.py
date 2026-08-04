"""
CSV -> RDF/XML Converter
========================
Converts three CSV files directly into OWL/RDF XML without requiring Neo4j.

Input files (same format as the Neo4j import script):
  - entity_type.csv   : Entity type definitions
  - relation_type.csv : Relation type definitions
  - verified_kg.csv   : Triple data

Output: OWL/RDF XML, directly openable in Protege or queryable via SPARQL

Install dependency:
  pip install rdflib
"""

import csv
import ast
import json
import re
import urllib.parse
from rdflib import Graph as RDFGraph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD


# =========================================================
# Configuration — modify as needed
# =========================================================
ENTITY_TYPE_FILE   = '../experimental results/RQ3/Java/entity_type.csv'
RELATION_TYPE_FILE = '../experimental results/RQ3/Java/relation_type.csv'
TRIPLE_FILE        = '../experimental results/RQ3/Java/verified_kg.csv'

BASE_URI    = "http://example.org/java-kg/"
OUTPUT_FILE = "java_kg.owl"
# =========================================================


def safe_uri(base: str, local: str) -> URIRef:
    """Safely encode a name as a URI fragment."""
    encoded = urllib.parse.quote(str(local).strip(), safe="")
    return URIRef(base + encoded)


def detect_encoding(file_path: str) -> str:
    """Auto-detect file encoding (uses chardet if available, falls back to gbk)."""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        enc = result.get('encoding') or 'gbk'
        print(f"  [encoding] {file_path} -> detected: {enc}")
        return enc
    except ImportError:
        print("  [encoding] chardet not installed, falling back to gbk")
        return 'gbk'


def load_entity_types(file_path: str) -> dict:
    """Load entity type definitions from CSV."""
    entity_types = {}
    with open(file_path, 'r', encoding=detect_encoding(file_path), errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            entity_type = row[0].strip().lower()
            description = row[1].strip().lower()
            if entity_type:
                standard_type = entity_type[0].upper() + entity_type[1:]
                entity_types[entity_type] = {
                    'label': standard_type,
                    'def':   description
                }
    print(f"  Loaded {len(entity_types)} entity types: {list(entity_types.keys())}")
    return entity_types


def load_relation_types(file_path: str) -> dict:
    """Load relation type definitions from CSV."""
    relation_types = {}
    with open(file_path, 'r', encoding=detect_encoding(file_path), errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            relation_type = row[0].strip().lower()
            description   = row[1].strip().lower()
            if relation_type:
                standard_type = relation_type.upper()
                relation_types[relation_type] = {
                    'type': standard_type,
                    'def':  description
                }
    print(f"  Loaded {len(relation_types)} relation types: {list(relation_types.keys())}")
    return relation_types


def safe_parse_triples(triples_str: str) -> list:
    """Attempt to parse a triples string using multiple fallback strategies."""
    if not triples_str or triples_str.strip() == '[]':
        return []
    try:
        return ast.literal_eval(triples_str)
    except:
        pass
    try:
        return json.loads(triples_str.replace("'", '"'))
    except:
        pass
    try:
        return re.findall(r'\(.*?\):\s*\(.*?\)', triples_str)
    except:
        pass
    return []


def parse_triple(triple_str: str) -> dict | None:
    """
    Parse a single triple string of the form:
      (head_type, relation_type, tail_type): (head_instance, relation_instance, tail_instance)
    Returns a dict or None if parsing fails.
    """
    try:
        parts = triple_str.split('): (')
        if len(parts) != 2:
            return None
        type_part     = parts[0].strip('(').split(', ')
        instance_part = parts[1].strip(')').split(', ')
        if len(type_part) != 3 or len(instance_part) != 3:
            return None
        return {
            'head_type':         type_part[0].strip().lower(),
            'relation_type':     type_part[1].strip().lower(),
            'tail_type':         type_part[2].strip().lower(),
            'head_instance':     instance_part[0].strip().lower(),
            'relation_instance': instance_part[1].strip().lower(),
            'tail_instance':     instance_part[2].strip().lower()
        }
    except Exception:
        return None


def build_rdf(entity_types: dict, relation_types: dict, triple_file: str, base_uri: str) -> RDFGraph:
    """
    Build an RDF graph from entity/relation type definitions and triple CSV data.

    Structure:
      - Each entity type -> OWL Class
      - Each relation type -> OWL ObjectProperty
      - Each triple instance -> OWL NamedIndividual + reified RDF Statement
        (to preserve relation_instance and source_text as annotations)
    """
    g = RDFGraph()

    # Define namespaces
    KG   = Namespace(base_uri)
    CLS  = Namespace(base_uri + "class/")
    IND  = Namespace(base_uri + "individual/")
    PROP = Namespace(base_uri + "property/")
    g.bind("kg",   KG)
    g.bind("cls",  CLS)
    g.bind("ind",  IND)
    g.bind("prop", PROP)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)

    # Ontology header
    ont = URIRef(base_uri)
    g.add((ont, RDF.type,     OWL.Ontology))
    g.add((ont, RDFS.label,   Literal("Java API Knowledge Graph", lang="en")))
    g.add((ont, RDFS.comment, Literal("Generated from CSV triples via csv_to_rdf.py")))

    # Declare all OWL Classes from entity type definitions
    for key, info in entity_types.items():
        class_uri = safe_uri(base_uri + "class/", info['label'])
        g.add((class_uri, RDF.type,        OWL.Class))
        g.add((class_uri, RDFS.label,      Literal(info['label'])))
        g.add((class_uri, RDFS.comment,    Literal(info['def'])))
        g.add((class_uri, RDFS.subClassOf, OWL.Thing))

    # Declare all OWL ObjectProperties from relation type definitions
    for key, info in relation_types.items():
        prop_uri = safe_uri(base_uri + "property/", info['type'])
        g.add((prop_uri, RDF.type,     OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label,   Literal(info['type'])))
        g.add((prop_uri, RDFS.comment, Literal(info['def'])))

    # Process triple CSV
    imported         = 0
    skipped          = 0
    errors           = 0
    seen_individuals = {}  # instance label -> class_uri (for deduplication)

    with open(triple_file, 'r', encoding=detect_encoding(triple_file), errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue

            source_text = row[0].strip().lower()
            triples_str = row[1].strip()

            if not triples_str or triples_str == '[]':
                skipped += 1
                continue

            triples_list = safe_parse_triples(triples_str)
            if not triples_list:
                errors += 1
                continue

            for triple_str in triples_list:
                triple = parse_triple(triple_str)
                if not triple:
                    continue

                head_info = entity_types.get(triple['head_type'])
                tail_info = entity_types.get(triple['tail_type'])
                rel_info  = relation_types.get(triple['relation_type'])

                if not head_info or not tail_info or not rel_info:
                    errors += 1
                    continue

                head_uri = safe_uri(base_uri + "individual/", triple['head_instance'])
                tail_uri = safe_uri(base_uri + "individual/", triple['tail_instance'])
                prop_uri = safe_uri(base_uri + "property/",   rel_info['type'])

                # Declare NamedIndividual for head (only on first occurrence)
                if triple['head_instance'] not in seen_individuals:
                    class_uri = safe_uri(base_uri + "class/", head_info['label'])
                    g.add((head_uri, RDF.type,        OWL.NamedIndividual))
                    g.add((head_uri, RDF.type,        class_uri))
                    g.add((head_uri, RDFS.label,      Literal(triple['head_instance'])))
                    g.add((head_uri, PROP.entityType, Literal(head_info['label'])))
                    seen_individuals[triple['head_instance']] = class_uri

                # Declare NamedIndividual for tail (only on first occurrence)
                if triple['tail_instance'] not in seen_individuals:
                    class_uri = safe_uri(base_uri + "class/", tail_info['label'])
                    g.add((tail_uri, RDF.type,        OWL.NamedIndividual))
                    g.add((tail_uri, RDF.type,        class_uri))
                    g.add((tail_uri, RDFS.label,      Literal(triple['tail_instance'])))
                    g.add((tail_uri, PROP.entityType, Literal(tail_info['label'])))
                    seen_individuals[triple['tail_instance']] = class_uri

                # Assert the main triple
                g.add((head_uri, prop_uri, tail_uri))

                # Reification: attach relation_instance and source_text as annotations
                stmt = BNode()
                g.add((stmt, RDF.type,              RDF.Statement))
                g.add((stmt, RDF.subject,           head_uri))
                g.add((stmt, RDF.predicate,         prop_uri))
                g.add((stmt, RDF.object,            tail_uri))
                g.add((stmt, PROP.relationInstance, Literal(triple['relation_instance'])))
                if source_text:
                    g.add((stmt, PROP.sourceText,   Literal(source_text)))

                imported += 1

            if idx % 500 == 0:
                print(f"     Processed {idx} rows... (triples imported: {imported:,})")

    print(f"\n  Result -> imported: {imported:,} | skipped: {skipped} | errors: {errors}")
    return g


def main():
    print("\n" + "=" * 60)
    print("  CSV -> RDF/XML Converter")
    print("=" * 60)

    print("\n[1/3] Loading type definitions...")
    entity_types   = load_entity_types(ENTITY_TYPE_FILE)
    relation_types = load_relation_types(RELATION_TYPE_FILE)

    print("\n[2/3] Converting triples to RDF...")
    rdf_graph = build_rdf(entity_types, relation_types, TRIPLE_FILE, BASE_URI)
    print(f"  Total RDF triples in graph: {len(rdf_graph):,}")

    print(f"\n[3/3] Serializing to RDF/XML -> {OUTPUT_FILE} ...")
    rdf_graph.serialize(destination=OUTPUT_FILE, format="xml")
    print(f"  Saved: {OUTPUT_FILE}")

    # Summary statistics
    classes = sum(1 for _ in rdf_graph.triples((None, RDF.type, OWL.Class)))
    props   = sum(1 for _ in rdf_graph.triples((None, RDF.type, OWL.ObjectProperty)))
    indivs  = sum(1 for _ in rdf_graph.triples((None, RDF.type, OWL.NamedIndividual)))
    stmts   = sum(1 for _ in rdf_graph.triples((None, RDF.type, RDF.Statement)))

    print("\n" + "-" * 60)
    print("  Summary")
    print("-" * 60)
    print(f"  OWL Classes          : {classes}")
    print(f"  OWL ObjectProperties : {props}")
    print(f"  Named Individuals    : {indivs}")
    print(f"  Reified Statements   : {stmts}")
    print(f"  Total RDF triples    : {len(rdf_graph):,}")
    print(f"  Output file          : {OUTPUT_FILE}")
    print("-" * 60)
    print("\n  Next steps:")
    print(f"  - Protege : File -> Open -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()