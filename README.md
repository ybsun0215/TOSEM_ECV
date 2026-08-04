# Explore-Construct-Verify: Balancing Richness and Reliability in API Knowledge Graph Construction

> **Paper:** *Balancing Richness and Reliability: An Explore-Construct-Verify Framework for API Knowledge Graph Construction*  
> **Authors:** Yanbang Sun, Qing Huang, Zhenchang Xing, Xiaoxue Ren, Xiaohong Li, Junjie Wang, Huan Jin, Zhiping Liu
---

## Overview

This repository contains the implementation of the **Explore-Construct-Verify (ECV)** framework, a three-stage pipeline for constructing semantically rich and structurally reliable API Knowledge Graphs (KGs) from developer documentation using large language models.

The framework consists of three stages:

1. **KG Exploration** — Bottom-up schema discovery from seed documents using LLMs. Entity types and relation types are inductively extracted and fused from small, representative texts.
2. **KG Construction** — Schema-guided triple extraction from the full document corpus. The explored schema constrains the LLM to produce consistent, API-focused triples.
3. **KG Verification (Filtering)** — Association rule mining (support, confidence, lift) is applied to the constructed KG to detect and remove statistically unreliable type triples, with a human-in-the-loop threshold adjustment mechanism.

The constructed KG can then be imported into Neo4j and used to enhance LLM-based API recommendation.

---

## Repository Structure

```
Explore-Construct-Verify/
├── new_code/
│   ├── main.py                  # Entry point: runs all three ECV stages
│   ├── kg_exploration.py        # Stage 1: schema discovery from seed texts
│   ├── kg_construction.py       # Stage 2: schema-guided triple extraction
│   ├── kg_filtering.py          # Stage 3: association rule-based verification
│   └── util.py                  # LLM call utilities
├── prompt/
│   ├── kg_exploration/          # Prompts and few-shot examples for Stage 1
│   └── kg_construction/         # Prompts and few-shot examples for Stage 2
├── seed_data/
│   └── seed.csv                 # Seed documents used for schema exploration
├── all_data/
│   └── parsed.csv               # Full document corpus for KG construction
├── output/
│   ├── kg_exploration/          # Outputs of Stage 1 (entity/relation types)
│   ├── kg_construction/         # Outputs of Stage 2 (extracted triples)
│   └── kg_filtering/            # Outputs of Stage 3 (verified KG)
├── support4apirecom/
│   ├── import2neo4j.py          # Imports the verified KG into Neo4j
│   ├── KG+LLM.py                # KG-enhanced LLM API recommendation evaluation
│   └── Ground_truth.csv         # Ground truth for API recommendation evaluation
├── config.yaml                  # All configuration (API keys, paths, thresholds)
└── Appendix.pdf                 # Supplementary details on prompts and schema design
```

---

## KG Schema

The ECV framework does **not** use a fixed, predefined schema. Instead, the schema (entity types and relation types) is **inductively discovered** during Stage 1 (KG Exploration) from a small set of seed documents.

### Schema Structure

The KG stores two levels of knowledge:

| Level | Description | Example |
|-------|-------------|---------|
| **Type triple** | Semantic relation between entity types | `(function, Equivalence, function)` |
| **Instance triple** | Concrete API relation between entity instances | `(remove, like, removeAt)` |

Each triple is represented as `(entity_type, relation_type, entity_type): (entity_instance, relation_instance, entity_instance)`.

### Entity Types

Entity types are automatically discovered per domain and language. For the Java dataset, discovered entity types include (but are not limited to):

| Entity Type | Definition | Example Instance |
|-------------|-----------|-----------------|
| `class` | A Java class or abstract class | `ArrayList`, `HashMap` |
| `interface` | A Java interface | `List`, `Iterable` |
| `method` | A callable method or function on an object | `add()`, `pollfirst()` |
| `package` | A Java package grouping related classes | `java.util` |

> **Note:** Entity types vary across different languages and are discovered during the exploration stage based on their respective corpora. The full list of discovered entity types and their LLM-generated definitions for each dataset is saved to `output/kg_exploration/entity_type.csv` after Stage 1.
These definitions of entity types can be regarded as attributes of each node (i.e., entity instance).

### Relation Types

Relation types are also discovered automatically. Common relation types found across datasets include:

| Relation Type | Definition | Example |
|---------------|-----------|---------|
| `Equivalence` | Two API entities have the same or nearly identical functionality | `(remove, like, removeAt)` |
| `Difference` | Two entities differ in behavior or thread-safety | `(Vector, synchronized_version_of, ArrayList)` |
| `Performance` | Two entities differ in computational performance | `(LinkedList, slower_than, ArrayList)` |
| `Dependency` | One entity requires or uses another | `(Iterator, used_by, Collection)` |

> The full set of discovered relation types and definitions is saved to `output/kg_exploration/relation_type.csv`. These are dataset-specific and may vary across languages and documentation corpora.
These definitions of relation types can be regarded as attributes of each edge (i.e., relation instance).

### Interpreting a Triple

Consider the triple: `(class, execution, method): (deque, removing the first element, pollFirst)`

- `class` and `method` are **entity types** (discovered by Stage 1)
- `execution` is the **relation type** (meaning: a class *executes via* a method)
- `deque`, `removing the first element`, and `pollFirst` are **entity instances**
- The instance `removing the first element` is the **relation instance label** — a natural language description of *how* the execution is manifested in this specific case

This dual-level representation (type + instance) allows the KG to capture both abstract semantic patterns and concrete API-level knowledge.

---

## Installation

Installation

**Requirements:** Python 3.8+, pip

```
# Clone the repository
git clone https://github.com/ybsun0215/Explore-Construct-Verify.git
cd Explore-Construct-Verify

# Install all dependencies automatically
pip install -r requirements.txt
```
---

## Configuration

All settings are managed in `config.yaml`. Edit this file before running:

```yaml
# 1. Provide one or more OpenAI-compatible API keys (used in parallel for speed)
API_key_list:
  - "sk-YOUR_KEY_1"
  - "sk-YOUR_KEY_2"

# 2. Choose your LLM (default: GPT-4o)
llm_name: "gpt-4o"
# llm_name: "deepseek-chat"

# 3. Set file paths (defaults work with the provided data)
seed_file_path: "../seed_data/"     # Seed documents for Stage 1 exploration
all_file_path:  "../all_data/"      # Full corpus for Stage 2 construction

# 4. Verification thresholds (see Threshold Tuning below)
predefined_support:    0.005
predefined_confidence: 0.02
predefined_lift:       1.0

# 5. Neo4j connection (for API recommendation evaluation only)
neo4j_uri:      "bolt://localhost:7687"
neo4j_user:     "neo4j"
neo4j_password: "YOUR_PASSWORD"
neo4j_db:       "YOUR_DB_NAME"
```

---

## Running the Framework

### Step 1: Run the Full ECV Pipeline

From the `new_code/` directory:

```
cd new_code
python main.py
```

This runs all three stages sequentially:
- **Stage 1 (Exploration):** Reads seed files from `seed_data/`, discovers entity/relation types, saves schema to `output/kg_exploration/`
- **Stage 2 (Construction):** Reads all documents from `all_data/`, extracts triples guided by the discovered schema, saves to `output/kg_construction/`
- **Stage 3 (Filtering/Verification):** Applies association rule mining, saves verified KG to `output/kg_filtering/refine_kg.csv`

### Step 2 (Optional): Import into Neo4j

After the pipeline completes, import the KG into Neo4j for querying and downstream tasks:

```
cd support4apirecom
python import2neo4j.py
```

Make sure Neo4j is running and credentials in `config.yaml` are correct.

### Step 3 (Optional): Evaluate API Recommendation

```
cd support4apirecom
python KG+LLM.py
```

Results are saved to the path specified by `output_path` in `config.yaml`.

---

## Using Your Own Documents

To run ECV on a **new API documentation corpus**:

1. **Prepare seed data** (for Stage 1 exploration):
   - Place a small, representative subset of your documentation (e.g., 20–50 text snippets) in `seed_data/` as `.csv` files
   - Each row should contain one text chunk in the first column

2. **Prepare full corpus** (for Stage 2 construction):
   - Place the complete documentation corpus in `all_data/` as `.csv` files in the same format

3. **Update `config.yaml`**:
   - Set `seed_file_path` and `all_file_path` to point to your directories

4. **Run the pipeline**:
   ```
   cd new_code && python main.py
   ```

The schema (entity and relation types) will be re-discovered from scratch based on your seed data — **no manual schema definition is required**.

---

## Threshold Tuning (KG Verification)

The verification stage uses three association rule metrics to filter unreliable type triples:

| Metric | Meaning | Effect of increasing |
|--------|---------|---------------------|
| `predefined_support` | Minimum proportion of triples matching a type pattern | Removes rare patterns |
| `predefined_confidence` | Minimum conditional probability of a relation given entity types | Removes unreliable relations |
| `predefined_lift` | Minimum association strength beyond random chance | Removes spurious co-occurrences |

**Defaults** (`support=0.0035`, `confidence=0.03`, `lift=1.0`) are calibrated for the Java dataset. For smaller corpora (e.g., Kotlin, Go), you may need to **lower** these thresholds to avoid over-filtering.

To tune interactively, modify the values in `config.yaml` and re-run Stage 3 only:

```python
# In new_code/main.py, call only kg_filtering:
kg_filtering(universal_config)
```

---

## Datasets and Evaluation Data

The datasets used in the paper (Java, Kotlin, Go) and evaluation ground truth are available in this repository and as a permanent archive:

- **Seed data:** `seed_data/seed.csv`
- **Full corpus:** `all_data/parsed.csv`
- **API recommendation ground truth:** `support4apirecom/Ground_truth.csv`

---

## Exported KG Format

The verified KG is output as a CSV file (`output/kg_filtering/refine_kg.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| `text` | Source text chunk from which the triple was extracted |
| `entity_type_1` | Type of the subject entity |
| `entity_instance_1` | Subject entity instance |
| `relation_type` | Type of the relation |
| `relation_instance` | Specific relation label for this instance triple |
| `entity_type_2` | Type of the object entity |
| `entity_instance_2` | Object entity instance |
| `source_text` | Original document source |

The KG is stored in a structured CSV format that can be directly imported into Neo4j using the provided support4apirecom/import2neo4j.py script. For users requiring RDF/OWL format or triple store ingestion, the support4apirecom/exportrdf.py script exports the KG as an OWL file, which can be ingested by standard triple stores such as RDF4J or Jena.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For questions about the code or data, please open a GitHub Issue or contact the author: ybsun@tju.edu.cn 
