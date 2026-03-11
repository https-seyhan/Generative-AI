Why Use spaCy Instead of Regex

### Regular expressions often struggle with:

Multiline SQL blocks

Inconsistent spacing

Comments

SAS macro expansions

### spaCy enables:

Token-level parsing

Context detection

Extensible pattern matching

Future ML-based classification of SAS statements

You can also extend the matcher to detect SAS operations such as:

SET
MERGE
JOIN
CREATE TABLE
INSERT INTO
UPDATE
Enterprise-Scale Architecture (Recommended)

For large SAS estates (10,000+ programs), a production scanner often uses a modular architecture.

SAS Scanner
   │
   ├── File crawler
   ├── spaCy parser
   ├── Table lineage builder
   ├── Dependency graph
   └── Metadata export

Example output schema:

file	library	table	operation	type
aml_job.sas	prod	transactions	set	source
aml_job.sas	ref	customer	join	source
aml_job.sas	risk	alerts	create	target

This approach enables:

SAS → Python migration discovery

ETL lineage documentation

Data governance mapping

Impact analysis for reporting pipelines

Possible Extensions

For enterprise environments, consider adding:

Parallel scanning for large repositories

Comment removal to reduce false positives

SAS macro parsing

Dependency graph generation using Graphviz

Data lineage export to governance tools

Summary

Using spaCy provides a more scalable and maintainable way to analyse SAS codebases compared to regex-only approaches. The method above allows automated extraction of table dependencies, forming the foundation for data lineage analysis, ETL migration, and governance documentation.
