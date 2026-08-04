import pandas as pd
from py2neo import Graph
import openai
import os
import json
import yaml
import argparse
from typing import List, Dict
import re


# ==================== LLM Call ====================
def call_llm(
        messages: list,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 4096,
):
    """Call an LLM to generate a response."""
    openai.api_key = api_key

    response = openai.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    return None


# ==================== API Name Utilities ====================
def normalize_api_name(api_name: str) -> str:
    """Normalize an API name for comparison."""
    if not api_name:
        return ""
    api_name = api_name.strip().lower()
    api_name = re.sub(r'\(.*?\)', '', api_name).strip()
    return api_name


def is_api_match(ground_truth: str, recommendation: str) -> bool:
    """Check whether the recommended API matches the ground truth."""
    gt = normalize_api_name(ground_truth)
    rec = normalize_api_name(recommendation)

    if not gt or not rec:
        return False

    # Exact match
    if gt == rec:
        return True

    # Bidirectional containment
    if gt in rec or rec in gt:
        return True

    # Core name match (last segment after '.' or '#')
    def core(name: str) -> str:
        parts = re.split(r'[.#]', name)
        return parts[-1] if parts else name

    gt_core = core(gt)
    rec_core = core(rec)

    if gt_core == rec_core and len(gt_core) > 2:
        return True

    if (gt_core in rec_core or rec_core in gt_core) and min(len(gt_core), len(rec_core)) > 2:
        return True

    return False


# ==================== AutoKGLLM Class ====================
class AutoKGLLM:
    """Automated KG+LLM pipeline for API recommendation."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, neo4j_db: str):
        """
        Initialize the KG+LLM system.

        Args:
            neo4j_uri:      Neo4j connection URI
            neo4j_user:     Neo4j username
            neo4j_password: Neo4j password
            neo4j_db:       Target database name
        """
        print(f"[INFO] Connecting to Neo4j database: {neo4j_db}...")
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password), name=neo4j_db)
        print("[INFO] Connected to Neo4j successfully!")

        try:
            count = self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
            print(f"[INFO] Total nodes in database: {count}")
        except Exception as e:
            print(f"[WARNING] Failed to count nodes: {e}")

    # ------------------------------------------------------------------
    # Step 1: Extract API entities from the question
    # ------------------------------------------------------------------
    def extract_api_entities(
            self,
            question: str,
            body: str,
            api_key: str,
            model_name: str = "gpt-4o"
    ) -> List[str]:
        """
        Extract API entities mentioned in the question.

        Returns:
            List of API entity strings (FQN and simple names).
        """
        prompt = f"""You are an expert in identifying API entities from natural language questions.

**Task**: Extract ALL API entities (refers to interfaces that are defined in the official API documentation) mentioned in the question.

**Question**: {question}

**Instructions**:
1. Extract all API entities (such as classes, methods, packages, etc.) existed in the question.
2. Return the result as a JSON list of strings
3. If no API entities found, return an empty list

**Examples**:
Input: java.util.Vector alternatives
Output: ["java.util.Vector"]

Input: How do I sum all the items of a list of integers in Kotlin?
Output: ["list"]

Input: How do we remove elements from a MutableList in Kotlin
Output: ["MutableList"]

Input: Create a io.Reader from a local file
Output: ["io.Reader"]

Input: In Go 1.18 strings.Title() is deprecated. What to use now? And how?
Output: ["strings.Title"]

Input: Go interface with String() method
Output: ["String"]

**Output Format**: JSON list only, no explanation
["entity1", "entity2", ...]

**Your Output**:"""

        messages = [
            {"role": "system", "content": "You are an expert API entity extraction assistant."},
            {"role": "user", "content": prompt}
        ]

        response = call_llm(messages, api_key, model_name)

        try:
            entities = json.loads(response)
        except json.JSONDecodeError:
            print(f"[WARNING] Failed to parse JSON, falling back to regex extraction...")
            print(f"[DEBUG] Raw response: {response}")
            pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z0-9]+)*\b'
            entities = re.findall(pattern, question + " " + body)

        # Expand entities with variants
        expanded = set()
        for entity in entities:
            expanded.add(entity)

            # Strip parentheses from method calls
            if '(' in entity:
                entity = re.sub(r'\(.*?\)', '', entity)
                expanded.add(entity)

            # Add simple name from FQN (e.g. io.Reader -> Reader)
            if '.' in entity:
                expanded.add(entity.split('.')[-1])

            # Handle hash-separated method references
            if '#' in entity:
                for part in entity.split('#'):
                    if part:
                        expanded.add(part)
                        if '.' in part:
                            expanded.add(part.split('.')[-1])

        entities = [e for e in expanded if len(e) >= 2]
        entities = list(set(filter(None, entities)))

        print(f"[INFO] Extracted API entities ({len(entities)} total): {entities}")
        return entities

    # ------------------------------------------------------------------
    # Step 2: Retrieve relevant triples from Neo4j
    # ------------------------------------------------------------------
    def retrieve_kg_triples(
            self,
            api_entities: List[str],
            max_triples: int = 50
    ) -> List[Dict]:
        """Retrieve triples from Neo4j for the given API entities (case-insensitive)."""
        print(f"\n[INFO] Retrieving triples from Neo4j for entities: {api_entities}")

        all_triples = []

        for entity in api_entities:
            variants = set()
            variants.add(entity)
            if '.' in entity:
                variants.add(entity.split('.')[-1])
            if '#' in entity:
                for p in entity.split('#'):
                    if p:
                        variants.add(p)
            variants.add(entity.replace('(', '').replace(')', ''))

            print(f"[DEBUG] Searching for entity '{entity}' with variants: {variants}")

            per_variant_limit = max(1, max_triples // (2 * len(variants)))

            for search_entity in variants:
                cypher_outgoing = """
                MATCH (h)-[r]->(t)
                WHERE toLower(h.entity_instance) = toLower($entity_name)
                RETURN
                    h.entity_instance as head,
                    h.entity_type as head_type,
                    type(r) as relation,
                    r.relation_type as relation_type,
                    r.relation_instance as relation_instance,
                    t.entity_instance as tail,
                    t.entity_type as tail_type
                LIMIT $limit
                """

                cypher_incoming = """
                MATCH (h)-[r]->(t)
                WHERE toLower(t.entity_instance) = toLower($entity_name)
                RETURN
                    h.entity_instance as head,
                    h.entity_type as head_type,
                    type(r) as relation,
                    r.relation_type as relation_type,
                    r.relation_instance as relation_instance,
                    t.entity_instance as tail,
                    t.entity_type as tail_type
                LIMIT $limit
                """

                try:
                    results_out = self.graph.run(
                        cypher_outgoing,
                        entity_name=search_entity.lower(),
                        limit=per_variant_limit
                    ).data()

                    results_in = self.graph.run(
                        cypher_incoming,
                        entity_name=search_entity.lower(),
                        limit=per_variant_limit
                    ).data()

                    if results_out or results_in:
                        print(f"[DEBUG] Found {len(results_out)} outgoing + {len(results_in)} "
                              f"incoming triples for '{search_entity}'")

                    for result in results_out + results_in:
                        triple = {
                            'head': result['head'],
                            'head_type': result.get('head_type', 'unknown'),
                            'relation': result['relation'],
                            'relation_type': result.get('relation_type', result['relation']),
                            'relation_instance': result.get('relation_instance', ''),
                            'tail': result['tail'],
                            'tail_type': result.get('tail_type', 'unknown'),
                        }
                        all_triples.append(triple)

                except Exception as e:
                    print(f"[WARNING] Failed to query entity variant '{search_entity}': {e}")
                    continue

        # Deduplicate
        seen = set()
        unique_triples = []
        for triple in all_triples:
            key = f"{triple['head']}|{triple['relation']}|{triple['tail']}"
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)

        print(f"[INFO] Retrieved {len(unique_triples)} unique triples from Neo4j")
        return unique_triples[:max_triples]

    # ------------------------------------------------------------------
    # Step 3: Filter relevant triples using LLM
    # ------------------------------------------------------------------
    def filter_relevant_triples(
            self,
            question: str,
            body: str,
            triples: List[Dict],
            api_key: str,
            top_k: int = 5,
            model_name: str = "gpt-4o"
    ) -> List[Dict]:
        """Select the top-k most relevant triples using an LLM."""
        print(f"\n[INFO] Filtering top-{top_k} relevant triples using LLM...")

        if not triples:
            print("[WARNING] No triples to filter!")
            return []

        if len(triples) <= top_k:
            print(f"[INFO] Number of triples ({len(triples)}) <= top_k ({top_k}), returning all")
            return triples

        triples_text = ""
        for i, triple in enumerate(triples):
            type_triple = f"({triple['head_type']}, {triple['relation_type']}, {triple['tail_type']})"
            rel_display = triple.get('relation_instance') or triple['relation_type']
            instance_triple = f"({triple['head']}, {rel_display}, {triple['tail']})"
            triples_text += f"[{i + 1}] {type_triple}: {instance_triple}\n\n"

        prompt = f"""You are an expert in API knowledge graph analysis. Your task is to select the most relevant triples for answering the given question.

**Question**: {question}

**Details**: {body}

**Available Knowledge Graph Triples**:
{triples_text}

**Task**:
1. Understand the user's needs based on the question and details.
2. Select the top {top_k} most semantically relevant triples that would help answer this question.
3. Return ONLY the indices (numbers) of the selected triples as a JSON list

**Examples**:
- If you select triples 1, 5, and 7, return: [1, 5, 7]
- If you select triples 2, 3, 10, 15, 20, return: [2, 3, 10, 15, 20]

**Output Format**: JSON list of integers only
[index1, index2, ..., index{top_k}]

**Your Selection**:"""

        messages = [
            {"role": "system", "content": "You are an expert at selecting relevant knowledge graph triples."},
            {"role": "user", "content": prompt}
        ]

        response = call_llm(messages, api_key, model_name, max_tokens=256)

        try:
            selected_indices = json.loads(response)
            selected_indices = [idx for idx in selected_indices if 1 <= idx <= len(triples)]
            selected_triples = [triples[idx - 1] for idx in selected_indices[:top_k]]
            print(f"[INFO] Selected {len(selected_triples)} triples: {selected_indices[:top_k]}")
            return selected_triples
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Failed to parse LLM response: {e}")
            print(f"[DEBUG] Raw response: {response}")
            print(f"[INFO] Fallback: returning first {top_k} triples")
            return triples[:top_k]

    # ------------------------------------------------------------------
    # Step 4: Generate the API recommendation
    # ------------------------------------------------------------------
    def generate_recommendation(
            self,
            question: str,
            body: str,
            selected_triples: List[Dict],
            api_key: str,
            model_name: str = "gpt-4o"
    ) -> str:
        """Generate an API recommendation based on the selected triples."""
        print(f"\n[INFO] Generating API recommendation using LLM...")

        kg_context = "**Knowledge Graph Information**:\n\n"
        for i, triple in enumerate(selected_triples, 1):
            type_triple = f"({triple['head_type']}, {triple['relation_type']}, {triple['tail_type']})"
            rel_display = triple.get('relation_instance') or triple['relation_type']
            instance_triple = f"({triple['head']}, {rel_display}, {triple['tail']})"
            kg_context += f"[Triple {i}] {type_triple}: {instance_triple}\n"

        kg_context += "\n"

        prompt = f"""You are an expert in API recommendation. Based on the provided knowledge graph triples, answer the following question by recommending a specific API.

**Question**: {question}

**Details**: {body}

{kg_context}

**Instructions**:
1. Analyze the question carefully
2. Carefully analyze the functionality of each retrieved API and distinguish their subtle differences.
3. Recommend ONE specific API that best answers the question
4. Your answer should be concise - just provide the API name (e.g., "ArrayList" or "SimpleDateFormat.format")

**Recommended API**:"""

        messages = [
            {"role": "system", "content": "You are an expert API recommendation assistant."},
            {"role": "user", "content": prompt}
        ]

        recommendation = call_llm(messages, api_key, model_name)
        print(f"[INFO] Generated recommendation: {recommendation}")
        return recommendation

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def recommend_api_automated(
            self,
            question: str,
            body: str,
            api_key: str,
            top_k_triples: int = 5,
            max_retrieve: int = 50,
            model_name: str = "gpt-4o",
            verbose: bool = True
    ) -> Dict:
        """
        Run the full automated KG+LLM recommendation pipeline.

        Returns:
            Dict with question, extracted entities, retrieved/selected triples, and recommendation.
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Question: {question}")
            print(f"{'=' * 80}")

        # Step 1: Entity extraction
        api_entities = self.extract_api_entities(question, body, api_key, model_name)

        if not api_entities:
            print("[WARNING] No API entities extracted, proceeding with empty KG context")
            retrieved_triples = []
            selected_triples = []
        else:
            # Step 2: Triple retrieval
            retrieved_triples = self.retrieve_kg_triples(api_entities, max_retrieve)

            # Step 3: Triple filtering
            if retrieved_triples:
                selected_triples = self.filter_relevant_triples(
                    question, body, retrieved_triples, api_key, top_k_triples, model_name
                )
            else:
                print("[WARNING] No triples retrieved from Neo4j")
                selected_triples = []

        # Step 4: Recommendation generation
        recommendation = self.generate_recommendation(
            question, body, selected_triples, api_key, model_name
        )

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Final Recommendation: {recommendation}")
            print(f"{'=' * 80}\n")

        return {
            'question': question,
            'body': body,
            'extracted_entities': api_entities,
            'retrieved_triples_count': len(retrieved_triples),
            'selected_triples': selected_triples,
            'recommendation': recommendation
        }


# ==================== Batch Evaluation ====================
def evaluate_auto_kg_llm(
        ground_truth_path: str,
        api_key: str,
        output_path: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_db: str,
        top_k_triples: int = 5,
        max_retrieve: int = 50,
        model_name: str = "gpt-4o"
):
    """
    Batch-evaluate the automated KG+LLM baseline.

    Args:
        ground_truth_path: Path to the ground truth CSV.
        api_key:           OpenAI API key.
        output_path:       Path to save the result CSV.
        neo4j_uri:         Neo4j URI.
        neo4j_user:        Neo4j username.
        neo4j_password:    Neo4j password.
        neo4j_db:          Neo4j database name.
        top_k_triples:     Number of triples to pass to the final LLM call.
        max_retrieve:      Maximum triples to retrieve from Neo4j.
        model_name:        LLM model identifier.
    """
    kg_llm = AutoKGLLM(neo4j_uri, neo4j_user, neo4j_password, neo4j_db)

    print(f"\n[INFO] Loading ground truth from {ground_truth_path}...")
    gt_df = pd.read_csv(ground_truth_path)

    # Flexible column name resolution
    column_mapping = {}
    possible_names = {
        'question':     ['Question', 'question', 'QUESTION', 'title', 'Title'],
        'body':         ['Body', 'body', 'BODY', 'description', 'Description'],
        'ground_truth': ['Ground_truth', 'ground_truth', 'GROUND_TRUTH', 'answer', 'Answer'],
        'language':     ['Language', 'language', 'LANGUAGE', 'lang', 'Lang'],
        'num':          ['Num', 'num', 'NUM', 'id', 'ID']
    }
    for key, candidates in possible_names.items():
        for col in candidates:
            if col in gt_df.columns:
                column_mapping[key] = col
                break

    print(f"[INFO] Column mapping: {column_mapping}")

    def format_triple(triple: Dict) -> str:
        if not triple:
            return ""
        type_triple = f"({triple['head_type']}, {triple['relation_type']}, {triple['tail_type']})"
        rel_display = triple.get('relation_instance') or triple['relation_type']
        instance_triple = f"({triple['head']}, {rel_display}, {triple['tail']})"
        return f"{type_triple}: {instance_triple}"

    results = []
    correct = 0
    total = len(gt_df)

    print(f"\n[INFO] Starting evaluation on {total} questions...\n")

    for idx, row in gt_df.iterrows():
        try:
            question     = row[column_mapping['question']]
            body         = row[column_mapping['body']]
            ground_truth = row[column_mapping['ground_truth']]
            language     = row.get(column_mapping.get('language', 'Language'), 'Unknown')
            num          = row.get(column_mapping.get('num', 'Num'), idx)
        except KeyError as e:
            print(f"[ERROR] Missing column: {e}")
            continue

        print(f"\n{'#' * 80}")
        print(f"Processing {idx + 1}/{total}")
        print(f"{'#' * 80}")

        try:
            result = kg_llm.recommend_api_automated(
                question=question,
                body=body,
                api_key=api_key,
                top_k_triples=top_k_triples,
                max_retrieve=max_retrieve,
                model_name=model_name,
                verbose=True
            )

            recommendation       = result['recommendation']
            selected_triples_list = result['selected_triples']

            is_correct = is_api_match(ground_truth, recommendation)
            if is_correct:
                correct += 1

            status = '✓ CORRECT' if is_correct else '✗ WRONG'
            print(f"[CHECK] GT: '{ground_truth}' vs REC: '{recommendation}' -> {status}")

            if selected_triples_list:
                print(f"[INFO] Using {len(selected_triples_list)} selected triples:")
                for i, t in enumerate(selected_triples_list, 1):
                    print(f"  [{i}] {format_triple(t)}")

            results.append({
                'Language':                   language,
                'Num':                        num,
                'Question':                   question,
                'Body':                       body,
                'Ground_Truth':               ground_truth,
                'Auto_KG_LLM_Recommendation': recommendation,
                'Is_Correct':                 is_correct,
                'Extracted_Entities':         ', '.join(result['extracted_entities']),
                'Retrieved_Triples_Count':    result['retrieved_triples_count'],
                'Selected_Triples_Count':     len(selected_triples_list),
                'Selected_Triple_1': format_triple(selected_triples_list[0]) if len(selected_triples_list) > 0 else '',
                'Selected_Triple_2': format_triple(selected_triples_list[1]) if len(selected_triples_list) > 1 else '',
                'Selected_Triple_3': format_triple(selected_triples_list[2]) if len(selected_triples_list) > 2 else '',
                'Selected_Triple_4': format_triple(selected_triples_list[3]) if len(selected_triples_list) > 3 else '',
                'Selected_Triple_5': format_triple(selected_triples_list[4]) if len(selected_triples_list) > 4 else '',
            })

            current_accuracy = correct / (idx + 1)
            print(f"\n[PROGRESS] Current Accuracy: {correct}/{idx + 1} = {current_accuracy:.2%}")

        except Exception as e:
            print(f"[ERROR] Failed to process question {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    final_accuracy = correct / len(results) if results else 0

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS — Automated KG+LLM")
    print(f"{'=' * 80}")
    print(f"Total Questions : {total}")
    print(f"Processed       : {len(results)}")
    print(f"Correct         : {correct}")
    print(f"Accuracy        : {final_accuracy:.2%}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")

    return results_df, final_accuracy


# ==================== Entry Point ====================
def main():
    # =========================================================
    # Load configuration from YAML file
    # Usage: python kg_llm_eval.py --config kg_llm_config.yaml
    # =========================================================
    parser = argparse.ArgumentParser(description="Automated KG+LLM Evaluation")
    parser.add_argument(
        "--config", type=str, default="../config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # =========================================================

    print("=" * 80)
    print("Automated KG+LLM Evaluation")
    print("=" * 80)

    results_df, accuracy = evaluate_auto_kg_llm(
        ground_truth_path = cfg["ground_truth_path"],
        api_key           = cfg["api_key"],
        output_path       = cfg["output_path"],
        neo4j_uri         = cfg["neo4j_uri"],
        neo4j_user        = cfg["neo4j_user"],
        neo4j_password    = cfg["neo4j_password"],
        neo4j_db          = cfg["neo4j_db"],
        top_k_triples     = cfg.get("top_k_triples", 5),
        max_retrieve      = cfg.get("max_retrieve", 50),
        model_name        = cfg.get("model_name", "gpt-4o"),
    )

    print(f"\n[SUCCESS] Automated KG+LLM Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()