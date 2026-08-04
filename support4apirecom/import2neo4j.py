from py2neo import Graph, Node, Relationship
import csv
import ast
import json
import os


def get_graph_connection(uri: str, user: str, password: str, db_name: str) -> Graph:
    """Connect to the Neo4j database."""
    return Graph(uri, auth=(user, password), name=db_name)


def complete_clean(graph: Graph):
    """Completely wipe the database (all nodes, relationships, constraints, and indexes)."""
    print("=" * 70)
    print("Clearing the database...")
    print("=" * 70)

    # 1. Delete all data
    graph.run("MATCH (n) DETACH DELETE n")
    print("✓ All nodes and relationships deleted")

    # 2. Drop all constraints
    try:
        constraints = graph.run("SHOW CONSTRAINTS").data()
        for constraint in constraints:
            name = constraint.get('name')
            if name:
                graph.run(f"DROP CONSTRAINT {name} IF EXISTS")
                print(f"✓ Dropped constraint: {name}")
    except:
        pass

    # 3. Drop all indexes
    try:
        indexes = graph.run("SHOW INDEXES").data()
        for index in indexes:
            name = index.get('name')
            if name and 'constraint' not in name.lower():
                try:
                    graph.run(f"DROP INDEX {name} IF EXISTS")
                    print(f"✓ Dropped index: {name}")
                except:
                    pass
    except:
        pass

    print("✓ Database fully cleared\n")


def load_entity_types(file_path: str) -> dict:
    """Load entity types from CSV file (normalized to lowercase)."""
    entity_types = {}

    print(f"[DEBUG] Reading entity type file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue

            # Normalize to lowercase
            entity_type = row[0].strip().lower()
            description = row[1].strip().lower()

            if entity_type:
                # Capitalize first letter as Neo4j label convention
                standard_type = entity_type[0].upper() + entity_type[1:]
                entity_types[entity_type] = {
                    'label': standard_type,
                    'def': description
                }
                print(f"  [DEBUG] Loaded entity type: '{entity_type}' -> label: '{standard_type}'")

    print(f"[INFO] Entity type keys: {list(entity_types.keys())}\n")
    return entity_types


def load_relation_types(file_path: str) -> dict:
    """Load relation types from CSV file (normalized to lowercase)."""
    relation_types = {}

    print(f"[DEBUG] Reading relation type file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue

            # Normalize to lowercase
            relation_type = row[0].strip().lower()
            description = row[1].strip().lower()

            if relation_type:
                # Uppercase as Neo4j relation type convention
                standard_type = relation_type.upper()
                relation_types[relation_type] = {
                    'type': standard_type,
                    'def': description
                }
                print(f"  [DEBUG] Loaded relation type: '{relation_type}' -> type: '{standard_type}'")

    print(f"[INFO] Relation type keys: {list(relation_types.keys())}\n")
    return relation_types


def parse_triple(triple_str: str) -> dict | None:
    """Parse a triple string into a dict (normalized to lowercase, with enhanced error handling)."""
    try:
        parts = triple_str.split('): (')
        if len(parts) != 2:
            return None

        type_part = parts[0].strip('(').split(', ')
        instance_part = parts[1].strip(')').split(', ')

        if len(type_part) != 3 or len(instance_part) != 3:
            return None

        # Normalize to lowercase
        return {
            'head_type': type_part[0].strip().lower(),
            'relation_type': type_part[1].strip().lower(),
            'tail_type': type_part[2].strip().lower(),
            'head_instance': instance_part[0].strip().lower(),
            'relation_instance': instance_part[1].strip().lower(),
            'tail_instance': instance_part[2].strip().lower()
        }
    except Exception:
        return None


def safe_parse_triples(triples_str: str) -> list:
    """Safely parse a list of triples (handles special characters)."""
    if not triples_str or triples_str.strip() == '[]':
        return []

    # Method 1: Try ast.literal_eval
    try:
        return ast.literal_eval(triples_str)
    except:
        pass

    # Method 2: Try json.loads (replace single quotes with double quotes)
    try:
        json_str = triples_str.replace("'", '"')
        return json.loads(json_str)
    except:
        pass

    # Method 3: Manual extraction via regex
    try:
        import re
        pattern = r'\(.*?\):\s*\(.*?\)'
        matches = re.findall(pattern, triples_str)
        return matches
    except:
        pass

    return []


def import_triples(file_path: str, entity_types: dict, relation_types: dict, graph: Graph):
    """Import triples from CSV into Neo4j (normalized to lowercase, with enhanced error handling)."""
    print("Starting triple import...\n")

    imported = 0
    skipped = 0
    errors = 0
    type_not_found = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        print(f"[DEBUG] CSV columns: {header}")

        for idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                continue

            # Normalize to lowercase
            source_text = row[0].strip().lower()
            triples_str = row[1].strip()

            if not triples_str or triples_str == '[]':
                skipped += 1
                continue

            triples_list = safe_parse_triples(triples_str)

            if not triples_list:
                errors += 1
                if idx % 100 == 0 or errors < 20:
                    print(f"  [ERROR] Row {idx} parse failed: {triples_str[:100]}...")
                continue

            for triple_str in triples_list:
                triple = parse_triple(triple_str)
                if not triple:
                    continue

                # Look up type info (keys are already lowercase)
                head_info = entity_types.get(triple['head_type'])
                tail_info = entity_types.get(triple['tail_type'])
                rel_info = relation_types.get(triple['relation_type'])

                # Track missing types
                if not head_info:
                    type_not_found[triple['head_type']] = type_not_found.get(triple['head_type'], 0) + 1
                if not tail_info:
                    type_not_found[triple['tail_type']] = type_not_found.get(triple['tail_type'], 0) + 1
                if not rel_info:
                    type_not_found[triple['relation_type']] = type_not_found.get(triple['relation_type'], 0) + 1

                if not head_info or not tail_info or not rel_info:
                    errors += 1
                    continue

                # Use entity_instance as the unique identifier for MERGE
                try:
                    cypher_create = f"""
                    MERGE (h:{head_info['label']} {{entity_instance: $head_instance}})
                    ON CREATE SET h.entity_type = $head_type, h.entity_type_def = $head_def
                    ON MATCH SET h.entity_type = $head_type, h.entity_type_def = $head_def
                    MERGE (t:{tail_info['label']} {{entity_instance: $tail_instance}})
                    ON CREATE SET t.entity_type = $tail_type, t.entity_type_def = $tail_def
                    ON MATCH SET t.entity_type = $tail_type, t.entity_type_def = $tail_def
                    MERGE (h)-[r:{rel_info['type']}]->(t)
                    ON CREATE SET 
                        r.relation_type = $rel_type, 
                        r.relation_type_def = $rel_def, 
                        r.relation_instance = $rel_instance,
                        r.source_text = $source
                    ON MATCH SET 
                        r.relation_type = $rel_type, 
                        r.relation_type_def = $rel_def, 
                        r.relation_instance = $rel_instance,
                        r.source_text = $source
                    """

                    graph.run(cypher_create,
                              head_instance=triple['head_instance'],
                              head_type=head_info['label'],
                              head_def=head_info['def'],
                              tail_instance=triple['tail_instance'],
                              tail_type=tail_info['label'],
                              tail_def=tail_info['def'],
                              rel_type=rel_info['type'],
                              rel_def=rel_info['def'],
                              rel_instance=triple['relation_instance'],
                              source=source_text)

                    imported += 1

                except Exception as e:
                    errors += 1
                    if errors < 20:
                        print(f"  [ERROR] Row {idx} import failed: {e}")

            if idx % 100 == 0:
                print(f"  Processed {idx} rows... (imported: {imported}, skipped: {skipped}, errors: {errors})")

    print(f"\n✓ Import complete!")
    print(f"  - Successfully imported: {imported} triples")
    print(f"  - Skipped empty rows:    {skipped}")
    print(f"  - Errors:                {errors}")

    if type_not_found:
        print(f"\n⚠️  The following types were not found in the type definitions (occurrence count):")
        for type_name, count in sorted(type_not_found.items(), key=lambda x: -x[1])[:10]:
            print(f"    - '{type_name}': {count} time(s)")
        print(f"\nSuggestion: Check that entity type and relation type files include these types.")
    print()


def verify_properties(graph: Graph):
    """Verify that node and relationship properties match the expected schema."""
    print("=" * 70)
    print("Property Verification")
    print("=" * 70 + "\n")

    # Check node properties
    print("Node property keys:")
    node_keys_query = """
    MATCH (n)
    WITH DISTINCT keys(n) as all_keys
    UNWIND all_keys as key
    RETURN DISTINCT key
    ORDER BY key
    """
    node_keys = graph.run(node_keys_query).data()
    node_key_list = [k['key'] for k in node_keys]
    print(f"  {node_key_list}")

    expected_node_keys = {'entity_instance', 'entity_type', 'entity_type_def'}
    if set(node_key_list) != expected_node_keys:
        print("  ⚠️  Warning: Nodes have unexpected or missing properties!")
        print(f"      Expected: {expected_node_keys}")
        print(f"      Actual:   {set(node_key_list)}")
    else:
        print("  ✓ Node properties are correct\n")

    # Check relationship properties
    print("Relationship property keys:")
    rel_keys_query = """
    MATCH ()-[r]->()
    WITH DISTINCT keys(r) as all_keys
    UNWIND all_keys as key
    RETURN DISTINCT key
    ORDER BY key
    """
    rel_keys = graph.run(rel_keys_query).data()
    rel_key_list = [k['key'] for k in rel_keys]
    print(f"  {rel_key_list}")

    expected_rel_keys = {'relation_type', 'relation_type_def', 'relation_instance', 'source_text'}
    if set(rel_key_list) != expected_rel_keys:
        print("  ⚠️  Warning: Relationships have unexpected or missing properties!")
        print(f"      Expected: {expected_rel_keys}")
        print(f"      Actual:   {set(rel_key_list)}")
    else:
        print("  ✓ Relationship properties are correct\n")

    # Sample nodes
    print("Sample nodes (unified naming):")
    sample_nodes = graph.run("MATCH (n) RETURN properties(n) as props LIMIT 3").data()
    for node in sample_nodes:
        print(f"  {node['props']}\n")

    # Sample relationships
    print("Sample relationships (unified naming):")
    sample_rels = graph.run("MATCH ()-[r]->() RETURN properties(r) as props LIMIT 3").data()
    for rel in sample_rels:
        print(f"  {rel['props']}\n")


def main():
    # =========================================================
    # Configuration — edit these parameters before running
    # =========================================================
    NEO4J_URI      = "bolt://localhost:7687"
    NEO4J_USER     = "neo4j"
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_PASSWORD")
    NEO4J_DB       = "java"

    ENTITY_TYPE_FILE  = './Java/entity_type.csv'
    RELATION_TYPE_FILE = './Java/relation_type.csv'
    TRIPLE_FILE        = './Java/verified_kg.csv'
    # =========================================================

    print("\n" + "=" * 70)
    print(" " * 15 + "Knowledge Graph Builder (Unified Naming)")
    print("=" * 70 + "\n")

    # Connect to Neo4j
    graph = get_graph_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)

    # Step 1: Clear the database
    complete_clean(graph)

    # Step 2: Load entity types
    print("=" * 70)
    print("Loading Entity Types")
    print("=" * 70)
    entity_types = load_entity_types(ENTITY_TYPE_FILE)
    print(f"\n✓ Loaded {len(entity_types)} entity type(s)")
    for key, info in entity_types.items():
        print(f"  • key='{key}' -> label='{info['label']}'")
    print()

    # Step 3: Load relation types
    print("=" * 70)
    print("Loading Relation Types")
    print("=" * 70)
    relation_types = load_relation_types(RELATION_TYPE_FILE)
    print(f"\n✓ Loaded {len(relation_types)} relation type(s)")
    for key, info in relation_types.items():
        print(f"  • key='{key}' -> type='{info['type']}'")
    print()

    # Step 4: Create uniqueness constraints on entity_instance
    print("=" * 70)
    print("Creating Database Constraints")
    print("=" * 70)
    for info in entity_types.values():
        try:
            graph.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{info['label']}) REQUIRE n.entity_instance IS UNIQUE"
            )
            print(f"  ✓ {info['label']}")
        except Exception as e:
            print(f"  ✗ {info['label']}: {e}")
    print()

    # Step 5: Import triples
    print("=" * 70)
    print("Importing Triples")
    print("=" * 70)
    import_triples(TRIPLE_FILE, entity_types, relation_types, graph)

    # Step 6: Statistics
    print("=" * 70)
    print("Statistics")
    print("=" * 70)

    node_count = graph.run("MATCH (n) RETURN count(n) as c").data()[0]['c']
    rel_count  = graph.run("MATCH ()-[r]->() RETURN count(r) as c").data()[0]['c']

    print(f"\nTotal nodes:         {node_count:,}")
    print(f"Total relationships: {rel_count:,}\n")

    print("Node distribution:")
    node_dist = graph.run(
        "MATCH (n) RETURN labels(n)[0] as label, count(*) as c ORDER BY c DESC"
    ).data()
    for item in node_dist:
        print(f"  {item['label']:15s}: {item['c']:,}")

    print("\nRelationship distribution:")
    rel_dist = graph.run(
        "MATCH ()-[r]->() RETURN type(r) as type, count(*) as c ORDER BY c DESC"
    ).data()
    for item in rel_dist:
        print(f"  {item['type']:15s}: {item['c']:,}")
    print()

    # Step 7: Verify properties
    verify_properties(graph)

    # Done
    print("=" * 70)
    print(" " * 25 + "✓ Done!")
    print("=" * 70 + "\n")

    print("Verification queries (unified naming):")
    print("  1. View nodes:")
    print("     MATCH (n) RETURN n.entity_instance, n.entity_type, n.entity_type_def LIMIT 5")
    print("\n  2. View relationships:")
    print("     MATCH ()-[r]->() RETURN r.relation_instance, r.relation_type, r.relation_type_def, r.source_text LIMIT 5")
    print("\n  3. View full triples:")
    print("     MATCH (h)-[r]->(t) RETURN h.entity_instance, r.relation_instance, t.entity_instance LIMIT 10")
    print("\n  4. Query specific entity:")
    print("     MATCH (n) WHERE n.entity_instance = 'vector' RETURN n")
    print()


if __name__ == "__main__":
    main()
