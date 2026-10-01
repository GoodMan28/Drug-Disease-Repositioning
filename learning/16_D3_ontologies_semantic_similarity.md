# Unit D3 — Biomedical ontologies and semantic similarity

> Part of the self-contained course that accompanies the MV-HGAT drug-repositioning project.
> Track D (biology / chemistry), unit 3 of 4. Comes after D2 (cheminformatics) and before D4 (biomedical databases).

---

## 0. Front matter

**Prerequisites**

| Unit | What you need from it |
|---|---|
| A1 Python | dictionaries, sets, reading text files line by line, pandas basics |
| A3 Probability | probability of an event, logarithms, $-\log p$ as "surprise" |
| C1 Graph theory | directed graphs, paths, cycles, depth-first and breadth-first search, topological order |
| D1 Pharmacology & repositioning | what a disease, a phenotype and an indication are |
| D2 Cheminformatics | the Jaccard/Tanimoto idea of "shared over total" (reused here) |

**Estimated study time:** 12–15 hours (5 h reading, 4 h code, 4–6 h exercises).

**Software and data:** plain Python (no ontology library is needed) plus NumPy/pandas/SciPy from the project's `.venv`. The real ontology files are in the project: `data/raw/ontology/mondo.obo` (MONDO release 2026-09-01, 53 MB) and `data/raw/ontology/doid.obo` (Disease Ontology release 2026-09-30). Run the code blocks *in order in one Python session* started in the project root; later blocks reuse earlier names. The general parser of Section 11.1 reads MONDO in under ten seconds; the project's specialised parser takes about one second.

**Learning objectives.** After this unit you will be able to:

1. Define ontology, term, identifier (CURIE/IRI), label, synonym (with scope), definition, cross-reference, subset and obsolescence, and recognise each in a real OBO file.
2. Explain the `is_a` and `part_of` relations, the true path rule, and why biomedical ontologies are directed acyclic graphs (DAGs) rather than trees.
3. Compute ancestors, descendants, minimum/maximum depth, lowest common ancestors (LCA) and the most informative common ancestor (MICA) on a DAG, by hand and in code.
4. Parse an OBO file with 30 lines of Python, handling stanzas, tags, trailing modifiers, comments and obsolete terms; and say how OBO relates to OWL.
5. Describe MeSH (descriptors, tree numbers), the Disease Ontology, OMIM (MIM numbers and prefixes), MONDO (a merged ontology with equivalence cross-references) and the HPO, and explain how phenotype similarity such as MimMiner is built.
6. Map identifiers between ontologies with cross-references, recognising one-to-many mappings, prefix mismatches and obsolete targets.
7. Compute edge-based (path, Wu–Palmer), information-content (Resnik, Lin, Jiang–Conrath) and hybrid (Wang) similarities; derive Wang's method step by step, including its closed form on trees and the role of the weight $w$.
8. Aggregate term-level similarities into entity-level similarities with max, average and best-match average (BMA), and argue which to use.
9. Read `02_build_features.py::parse_mondo`, `disease_table` and `similarity.py::wang_s_values` / `wang_similarity` line by line, and reproduce the project's `sem_mondo` view.
10. Diagnose the classic pitfalls: shallow annotation, uneven depth, grouping classes, obsolete terms, root similarity, violations of the true path rule.

---

## 1. Motivation: describing diseases for a cold-start model

The project's hardest evaluation is **cold start**: a disease whose known drugs are all hidden (leave-one-disease-out). The model then knows nothing about the disease's treatments and must rely entirely on how the disease *resembles* other diseases. The disease side has three similarity views:

| View | Built from | Coverage |
|---|---|---|
| `pheno_mim` | the benchmark's MimMiner matrix: text-mined phenotype similarity of OMIM records (van Driel et al., 2006) | all diseases |
| `sem_mondo` | **ours**: Wang semantic similarity on the MONDO disease hierarchy | 97.1% (F), 97.3% (C) of diseases |
| `gene_d` | ours: Jaccard overlap of CTD curated disease genes | about half of the diseases |

Standalone signal (score each pair from the 10 nearest diseases in one view, no learning; `HOW_IT_WORKS.md`, Section 10):

| Cold start, disease side | Fdataset AUPR | Cdataset AUPR |
|---|---|---|
| `pheno_mim` | 0.149 | 0.318 |
| `sem_mondo` | 0.106 | 0.222 |
| `pheno_mim` + `sem_mondo` | **0.175** | **0.322** |

The ontology view is weaker alone than the phenotype view but *adds* to it: the two agree very little (Spearman correlation 0.05 over all Fdataset disease pairs; Section 11.10), so they carry different information. On Fdataset the combination improves AUPR by 17% over phenotype similarity alone.

A concrete example. OMIM 104300 is *Alzheimer disease type 1* (familial early-onset Alzheimer disease caused by *APP* mutations). Its five nearest Fdataset diseases in each view:

| `sem_mondo` (ontology) | `pheno_mim` (text-mined phenotype) |
|---|---|
| Alzheimer disease, familial early-onset (0.544) | Huntington disease (0.460) |
| Lewy body dementia (0.298) | Alzheimer disease, familial early-onset (0.419) |
| central diabetes insipidus (0.287) | schizophrenia (0.410) |
| early-onset ataxia with oculomotor apraxia (0.279) | ischemic stroke (0.370) |
| cerebral arteriopathy, autosomal dominant (0.279) | Unverricht–Lundborg syndrome (0.363) |

The ontology view finds the classification neighbours (another familial Alzheimer form, another dementia); the phenotype view finds diseases whose *descriptions* mention similar symptoms (dementia, psychosis, neuronal loss). One suspicious entry — diabetes insipidus — is a lesson in itself (Section 10). To build, trust and debug such a view you need to know what an ontology is, how to read one, and how semantic similarity is computed. That is this unit.

Why MONDO and not the Disease Ontology (DO) or MeSH directly? Our diseases are identified by OMIM numbers. Following OMIM cross-references, **MONDO places 97.1% (F) / 97.3% (C) of them** in its hierarchy; DO alone reaches only **59.1% / 64.5%** (Section 11.8). MONDO was built precisely to merge OMIM, Orphanet, DO, MeSH, NCIt and others into one hierarchy with explicit equivalences.

---

## 2. What an ontology is

### 2.1 From word lists to ontologies

It helps to see an ontology as the end of a spectrum of increasingly structured vocabularies:

1. **Controlled vocabulary** — an agreed list of terms ("Alzheimer disease", "dementia"), so that everyone writes the same string.
2. **Thesaurus** — terms plus synonyms and *broader/narrower* links ("dementia" is broader than "Alzheimer disease"). MeSH is a thesaurus.
3. **Taxonomy** — a hierarchy of classes linked by "is a kind of".
4. **Ontology** — a formal, machine-interpretable specification of the classes in a domain and of the *typed* relations between them (is_a, part_of, has_material_basis_in, …), with definitions precise enough for automatic reasoning. Classic definition (Gruber, 1993): "an explicit specification of a conceptualisation".

In biomedicine "ontology" is used loosely for everything from 2 to 4. What matters for us is that each resource provides **classes with stable identifiers arranged in a hierarchy**.

### 2.2 Anatomy of a term

Here is a real term from `mondo.obo` (abridged), the one OMIM 104300 maps to:

```
[Term]
id: MONDO:0007088
name: Alzheimer disease type 1
synonym: "AD1" EXACT ABBREVIATION [GARD:0009465, OMIM:104300]
synonym: "Alzheimer disease, familial, 1" EXACT [OMIM:104300]
synonym: "early-onset familial form of Alzheimer disease" BROAD [GARD:0009465]
xref: DOID:0080348 {source="MONDO:equivalentTo"}
xref: MESH:C536594 {source="MONDO:equivalentTo"}
xref: OMIM:104300 {source="MONDO:equivalentTo", source="DOID:0080348"}
xref: UMLS:C1863052 {source="MEDGEN:354892", source="MONDO:equivalentTo", source="MONDO:MEDGEN"}
is_a: MONDO:0015140 {source="https://www.ncbi.nlm.nih.gov/books/NBK1236/"} ! early-onset autosomal dominant Alzheimer disease
is_a: MONDO:1060190 {source="https://clinicalgenome.org/affiliation/40156/"} ! APP-related brain and vascular amyloidosis
```

The parts:

* **Identifier (ID).** `MONDO:0007088` is a **CURIE** (compact URI): a *prefix* (`MONDO`) naming the source, a colon, and a *local identifier*. The full web identifier (IRI) is obtained by expanding the prefix: OBO ontologies use `http://purl.obolibrary.org/obo/MONDO_0007088`. IDs are meaningless numbers on purpose: they never change when the name changes, so data annotated years ago stays valid.
* **Label** (`name:`) — the preferred human-readable name. Labels *can* change; never join on labels.
* **Synonyms** with a **scope**: `EXACT` (same meaning), `BROAD` (the synonym is broader than the term), `NARROW`, `RELATED`; optionally a type (`ABBREVIATION`) and the sources that use it. Synonyms are what text-mining and name matching use.
* **Definition** (`def:`) — a text definition with references (this term has none; its sibling MONDO:0007089 does).
* **Cross-references** (`xref:`) — identifiers of the "same" or related concept in other resources. In MONDO the trailing qualifier `{source="MONDO:equivalentTo"}` says the mapping is a curated *equivalence*, not just "related".
* **Parents** (`is_a:`) — the superclasses. This term has **two** parents, which is why the structure is a DAG (Section 3).
* **Subsets** (`subset:`) — tags grouping terms for particular uses ("rare", "otar", "ordo_disease").
* **Obsolescence** — terms are never deleted; they are marked `is_obsolete: true`, with `replaced_by:` (a direct substitute) or `consider:` (candidates) where appropriate. MONDO's release contains 4,618 obsolete stanzas, 3,906 of them MONDO's own classes (Section 11.1).

### 2.3 Classes, instances and annotations

An ontology term is a **class** (a kind of thing: "Alzheimer disease"), not an individual (a patient's disease). Data about real things — a patient has AD, a gene is associated with AD, a drug treats AD — are **annotations** that point to the class. Keep the distinction in mind for information content (Section 7.3): "how specific is a class" can be measured from the ontology's structure (intrinsic) or from how often the class is used in annotations (corpus-based).

---

## 3. Relations and graph structure

### 3.1 `is_a` and `part_of`

* **`is_a`** (subclass of): every instance of the child is an instance of the parent. "Alzheimer disease type 1 is_a familial Alzheimer disease" means every case of AD1 is a case of familial AD. `is_a` is **transitive** (if A is_a B and B is_a C, then A is_a C) and supports **inheritance**: whatever is true of all members of the parent is true of the child.
* **`part_of`** (BFO:0000050): every instance of the child is *part of* some instance of the parent ("mitochondrial matrix part_of mitochondrion"). It is transitive too, but it is not subsumption: a nucleus is not a kind of cell. In OBO files it appears as `relationship: part_of …`. Disease ontologies use few part-whole relations between diseases; MONDO uses other relations to link diseases to anatomy and genes (`disease_has_location`, `has_material_basis_in_germline_mutation_in`), which our project ignores.

Semantic-similarity methods usually use only `is_a`, sometimes `part_of` with a lower weight (Wang's original GO method: 0.8 for is_a, 0.6 for part_of).

### 3.2 The true path rule

The **true path rule** (from the Gene Ontology) says that annotating an entity to a term implies annotating it to *every ancestor* of that term along *every* path. If a disease is "Alzheimer disease type 1", it is also "familial Alzheimer disease", "Alzheimer disease", "dementia", "neurodegenerative disease", "nervous system disorder", and "disease". An ontology must be built so that this is always true — a child must never be placed under a parent that is false for some of its instances. Every ancestor-based similarity measure silently assumes the rule holds, so a violation (a wrong `is_a` edge) propagates into every similarity score that passes through it.

### 3.3 Trees versus DAGs

A **tree** has one root and every node has exactly one parent. Real biomedical hierarchies need **multiple inheritance**: Alzheimer disease type 1 is both a kind of early-onset autosomal dominant Alzheimer disease *and* a kind of APP-related amyloidosis. So the structure is a **directed acyclic graph** (DAG): edges point from child to parent, a node may have several parents, and there are no directed cycles (nothing can be its own ancestor). In the 2026-09-01 MONDO release, **10,722 of the 32,109 live MONDO classes (33%) have more than one parent** (Section 11.1).

Consequences of "DAG, not tree":

* There can be **several paths** from a term to the root, of **different lengths**, so "depth" is not unique.
* Two terms can have **several lowest common ancestors**.
* Algorithms must avoid visiting a node twice (or must deliberately take the max/min over paths, as Wang's method does).

### 3.4 Ancestors, descendants, depth, LCA and MICA

For a DAG with parent function $\mathrm{pa}(t)$:

* **Ancestors** $\mathcal{A}(t)$: all terms reachable by following parent edges, *including $t$ itself* (convention in this chapter; Wang calls this set $T_t$).
* **Descendants** $\mathcal{D}(t)$: all terms from which $t$ is reachable, including $t$.
* **Minimum depth**: length of the shortest path from $t$ to a root; **maximum depth**: length of the longest. Alzheimer disease type 1 has minimum depth 7 and maximum depth 10 in MONDO.
* **Common ancestors** $\mathcal{A}(a) \cap \mathcal{A}(b)$.
* **Lowest common ancestors (LCA)**: common ancestors none of whose descendants is also a common ancestor. In a tree there is exactly one; in a DAG there may be several. For AD1 and late-onset Parkinson disease MONDO has two LCAs: *brain disorder* and *hereditary neurological disease*.
* **Most informative common ancestor (MICA)**: the common ancestor with the largest information content (Section 7.3). It is always one of the LCAs (an ancestor of a common ancestor cannot be more specific), but which one depends on the IC.

A small DAG we will use throughout (edges point upward, child → parent):

```
                      R  "disease"
                    /    \
       "nervous"  N        M  "metabolic"
                  |         \
     "neurodeg."  D          \
                / | \         \
     "AD-like" A  P  W---------+          W is_a D  and  W is_a M
                  "PD-like"  "Wilson-like" (a disease that is both
                                            neurodegenerative and metabolic)
```

| Term | Parents | Ancestors (incl. self) | Descendants (incl. self) | min depth | max depth |
|---|---|---|---|---|---|
| R | – | R | all 7 | 0 | 0 |
| N | R | N, R | N, D, A, P, W | 1 | 1 |
| M | R | M, R | M, W | 1 | 1 |
| D | N | D, N, R | D, A, P, W | 2 | 2 |
| A | D | A, D, N, R | A | 3 | 3 |
| P | D | P, D, N, R | P | 3 | 3 |
| W | D, M | W, D, M, N, R | W | 2 (via M) | 3 (via D) |

LCA(A, P) = {D}. LCA(A, W) = {D}. LCA(W, M) = {M}. LCA(A, M) = {R}.

---

## 4. The OBO format, and OWL in brief

### 4.1 Structure of an OBO file

OBO (Open Biomedical Ontologies) flat-file format, version 1.2/1.4, is a line-oriented text format designed to be readable by humans and trivial to parse. A file has:

1. A **header**: tag–value lines before the first stanza.
   ```
   format-version: 1.2
   data-version: releases/2026-09-01
   subsetdef: rare "rare"
   synonymtypedef: ABBREVIATION "abbreviation"
   idspace: OMIM https://omim.org/entry/
   ```
   `idspace` lines tell you how to expand prefixes to IRIs; `subsetdef` and `synonymtypedef` declare the subset and synonym-type names used later.
2. **Stanzas**, each starting with a header in square brackets: `[Term]` (a class), `[Typedef]` (a relation type, e.g. `part_of` with `is_transitive: true`), or `[Instance]`. MONDO's file has 63,278 `[Term]` and 282 `[Typedef]` stanzas.
3. Inside a stanza, one **tag: value** pair per line. The common tags:

| Tag | Example value | Meaning |
|---|---|---|
| `id` | `MONDO:0007088` | identifier (exactly one) |
| `name` | `Alzheimer disease type 1` | label |
| `def` | `"text" [PMID:123, OMIM:104300]` | quoted definition + supporting references |
| `synonym` | `"AD1" EXACT ABBREVIATION [OMIM:104300]` | quoted text, scope, optional type, references |
| `xref` | `OMIM:104300 {source="MONDO:equivalentTo"}` | cross-reference |
| `is_a` | `MONDO:0015140 ! early-onset …` | parent class |
| `relationship` | `disease_has_location UBERON:0000955` | typed edge to another class |
| `intersection_of` | `MONDO:0000001`, `has_material_basis_in …` | logical definition (genus + differentia) |
| `subset` | `rare` | subset membership |
| `alt_id` | `DOID:0110051` | secondary ID merged into this term |
| `is_obsolete` | `true` | term retired |
| `replaced_by` / `consider` | `MONDO:0009299` | what to use instead |
| `property_value` | `skos:exactMatch OMIM:104300` | generic annotation |

4. Two syntactic decorations that parsers must handle:
   * **Trailing modifiers** in braces, `{source="…", …}`, qualify a value (provenance, mapping type).
   * **Comments** after `!` are for humans (`is_a: MONDO:0015140 ! early-onset …` — the label after `!` is ignored by machines). Inside quoted strings, `\"`, `\n` and `\!` are escapes.

### 4.2 How to parse it (algorithm)

```
terms = {}; current = None
for each line:
    strip the newline
    if line starts with "[":                # new stanza
        current = new dict if line == "[Term]" else None   (skip Typedefs)
    elif current is not None and ":" in line:
        tag, value = split at the first ": "
        drop a trailing "{...}" modifier and a trailing "! comment" if you do not need them
        append value to current[tag]
        if tag == "id": terms[value] = current
afterwards: drop obsolete terms; keep only the prefix you care about;
            build parents[t] = [p for p in is_a(t) if p is a kept term]
```

This is exactly what the project's `parse_mondo` does (Section 12), in a version specialised to the five tags it needs. Section 11.1 gives a general version that keeps every tag.

Two gotchas specific to MONDO's OBO release:

* **Imported terms.** `mondo.obo` also contains thousands of stanzas from *other* ontologies that MONDO's logical definitions reference: UBERON anatomy (5,360), HP phenotypes (4,859), GO (4,142), NCBITaxon, CHEBI, CL and more — only 36,015 of the 63,278 stanzas are MONDO's own (including obsolete ones). A parser that does not filter on the `MONDO:` prefix will happily mix anatomy and chemistry into the disease DAG.
* **Multiple roots.** Live MONDO has four root classes: *disease* (MONDO:0000001), *disease characteristic*, *injury*, and *disease susceptibility* (MONDO:0042489). Similarity between terms under different roots is 0 for Wang's method (no common ancestor). This is not academic for us: OMIM "susceptibility to …" entries map to classes under *disease susceptibility*, and 33 of Fdataset's 313 diseases (65 of Cdataset's 409) sit *only* under that root (Exercise 12).

Ready-made parsers exist — `pronto` and `obonet` (Python), the Ontology Access Kit (OAK) — and are worth using in larger projects. For one file and five tags, 25 lines of Python are clearer and have no dependencies.

### 4.3 OWL in one page

The **Web Ontology Language (OWL 2)** is the W3C standard for ontologies, grounded in description logic. OBO is now formally defined as a *serialisation* of a subset of OWL: every OBO construct has an OWL meaning.

| OBO | OWL |
|---|---|
| `[Term] id: X` | `Class: X` |
| `is_a: Y` | `X SubClassOf: Y` |
| `relationship: part_of Y` | `X SubClassOf: part_of some Y` (an existential restriction) |
| `intersection_of: Y` + `intersection_of: R Z` | `X EquivalentTo: Y and (R some Z)` |
| `disjoint_from: Y` | `DisjointClasses: X, Y` |
| `xref`, `synonym`, `def` | annotation properties (no logical meaning) |

The main practical difference: in OWL, **logical definitions** let a **reasoner** (ELK, HermiT) *infer* subclass links that nobody asserted. MONDO is developed in OWL; part of its hierarchy is inferred from definitions like "an Alzheimer disease that has material basis in a mutation in *APP*". The OBO release we parse already contains the inferred `is_a` edges, which is why simple OBO parsing is enough for similarity work. The full OWL release (`mondo.owl`) also contains equivalence axioms to other ontologies and is what you would load in Protégé.

---

## 5. Disease vocabularies you will meet

### 5.1 MeSH — Medical Subject Headings

MeSH is the U.S. National Library of Medicine's thesaurus for indexing the biomedical literature (every PubMed article is tagged with MeSH headings). Key concepts:

* **Descriptors** (main headings) have IDs `D` + 6 digits (`D000544` = Alzheimer Disease). **Supplementary Concept Records** (SCRs) for rare diseases and chemicals have IDs `C` + 6 or 9 digits (`C536594` = Alzheimer disease type 1). SCRs are not in the tree; they are linked to one or more descriptors ("heading mapped to"). CTD's copy of MeSH gives SCRs pseudo tree numbers by appending the SCR to its descriptor's tree number (`C10.228.140.380.100/C536594`).
* **Tree numbers.** Descriptors are arranged in 16 top-level categories (A anatomy, C diseases, D chemicals and drugs, F psychiatry and psychology, …). A descriptor's position is a dotted **tree number**, and its parent is obtained by deleting the last segment. A descriptor can appear in **several places**, so it can have several tree numbers. Alzheimer Disease (D000544) has three (from the CTD copy of MeSH in the project, Section 11.9):

```
C10.228.140.380.100   Alzheimer Disease   (under C10.228.140.380 Dementia, C10.228.140 Brain Diseases,
                                            C10.228 Central Nervous System Diseases, C10 Nervous System Diseases)
C10.574.945.249       Alzheimer Disease   (under C10.574.945 Tauopathies, C10.574 Neurodegenerative Diseases)
F03.615.400.100       Alzheimer Disease   (under F03.615.400 Dementia, F03.615 Neurocognitive Disorders,
                                            F03 Mental Disorders)
```

So MeSH, viewed as a graph of descriptors, is a DAG too, even though each tree number lives in a strict tree. Tree numbers make ancestor computation trivial (string prefixes), which is why many disease-similarity papers (Wang et al., 2010, for microRNA–disease work) used MeSH. Caveat: MeSH's hierarchy expresses "broader/narrower for indexing", not strict `is_a`; occasionally a child is not logically a subtype of its parent. Also, tree numbers are **not stable identifiers** — they change when the tree is reorganised; always store the `D` number.

### 5.2 The Disease Ontology (DO, DOID)

The Human Disease Ontology (Schriml et al.) is an OBO Foundry ontology of human diseases with IDs like `DOID:10652` (Alzheimer's disease, `is_a DOID:680 tauopathy`). It is organised mainly by anatomical location and etiology, has text definitions and rich cross-references (MeSH, ICD-10, ICD-9, NCI, SNOMED CT, UMLS, OMIM). The 2026-09-30 release in the project has 12,335 live terms. Its weakness for our use is coverage of rare Mendelian diseases: OMIM's 2011-era entries are often individual genetic subtypes ("Alzheimer disease 2", "Type 1 diabetes mellitus 11") that DO does not list separately. Following DO's OMIM cross-references reaches 59.1% of Fdataset's diseases and 64.5% of Cdataset's.

A trap waiting in the file: DO writes OMIM cross-references as **`MIM:104300`**, whereas MONDO writes **`OMIM:104300`**. A join on the literal prefix `OMIM:` finds *zero* DO mappings (we made that mistake first; Section 11.8). Always normalise prefixes (the Bioregistry, bioregistry.io, lists standard prefixes and their synonyms).

### 5.3 OMIM and MIM numbers

OMIM (Online Mendelian Inheritance in Man, Johns Hopkins University) is a curated catalogue of human genes and genetic phenotypes, descended from Victor McKusick's book *Mendelian Inheritance in Man* (first edition 1966). Each entry has a six-digit **MIM number**. The first digit encodes the entry's category:

| First digit | Meaning |
|---|---|
| 1, 2 | autosomal loci or phenotypes, entries created before 15 May 1994 (historically 1 = dominant, 2 = recessive) |
| 3 | X-linked |
| 4 | Y-linked |
| 5 | mitochondrial |
| 6 | autosomal, entries created after 15 May 1994 |

A symbol before the number gives the entry type: `*` gene; `#` phenotype with known molecular basis; `%` phenotype with Mendelian inheritance but unknown molecular basis; `+` (rare, historical) gene and phenotype combined; no symbol: phenotype with suspected Mendelian basis; `^` entry removed or moved. In phenotype names, braces `{…}` mark susceptibility to multifactorial disorders, brackets `[…]` "nondiseases" (e.g. lab-value variants), and `?` a provisional gene–phenotype relationship. **Phenotypic series** (IDs `PS` + six digits) group genetically heterogeneous forms of the same phenotype (e.g. all "Alzheimer disease" loci).

OMIM is a catalogue, **not an ontology**: entries are flat, with no `is_a` hierarchy. That is the core reason the project needs MONDO — to place each OMIM disease of the benchmark in a hierarchy. In the benchmark files the disease IDs are written `D104300`; the `D` is dropped to get the MIM number (`02_build_features.py::load_benchmark`).

### 5.4 MONDO — a merged disease ontology

**MONDO** (Mondo Disease Ontology, Monarch Initiative; Vasilevsky et al., 2022) was built because no single disease resource covered both common and rare diseases with a logically sound hierarchy, and different resources disagreed. MONDO integrates OMIM, Orphanet, DO, NCIt, MeSH, ICD-10/11, GARD, MedGen/UMLS and others, using a semi-automated probabilistic merging method plus manual curation. Its distinctive features:

* **Precise mappings.** Every cross-reference carries a qualifier: `MONDO:equivalentTo` (the source class means exactly this MONDO class), `MONDO:relatedTo`, and for retired classes `MONDO:obsoleteEquivalent` (an obsolete MONDO class that was equivalent to the source), `MONDO:equivalentObsolete` (the *source* entry is obsolete). In the 2026-09-01 release the `OMIM:` cross-references are 9,773 `equivalentTo`, 131 `equivalentObsolete`, 221 `obsoleteEquivalent` and 51 `obsoleteEquivalentObsolete`.
* **One-to-one equivalence policy.** An equivalence axiom says two classes are *the same*; if two MONDO classes were both equivalent to one OMIM entry, they would be logically equivalent to each other and should be merged. Hence, among live MONDO classes, **each OMIM number maps to at most one class** — and indeed, in this release none of the 9,904 OMIM IDs referenced by live MONDO classes maps to more than one (Section 11.8). The project code still handles several (Section 12, self-check Q2).
* **Grouping classes and multiple axes.** MONDO classifies diseases along several axes simultaneously: by organ system ("nervous system disorder"), by mechanism ("hereditary disease", "autosomal dominant disease", "disease by molecular mechanism") and by clinical grouping ("dementia"). This is the source of many multiple-parent terms — and of some surprising similarities (Section 10).
* **Obsolescence with provenance.** Diseases that OMIM or the literature no longer support are obsoleted, not deleted. Example: MONDO:0010582 *obsolete diabetes insipidus, neurohypophyseal type, X-linked inheritance*, with the comment that the last evidence dates from 1967 — which is why OMIM 304900 in our benchmark maps to no live MONDO class.

### 5.5 HPO and phenotype-based disease similarity

The **Human Phenotype Ontology** (HPO; Köhler, Robinson et al.) describes phenotypic *abnormalities* — signs, symptoms, lab findings (`HP:0000726` Dementia; `HP:0001250` Seizure) — under the root `HP:0000118 Phenotypic abnormality`. Diseases from OMIM and Orphanet are annotated with sets of HPO terms (the `phenotype.hpoa` file). Two diseases can then be compared by comparing their **phenotype term sets** with an ontology-aware similarity: Köhler et al. (2009, the *Phenomizer*) used Resnik similarity between HPO terms and aggregated over the two sets with a best-match approach. This is the modern, structured way to do "phenotype similarity", and it is used in rare-disease diagnosis.

The benchmark's `pheno_mim` view predates HPO's disease annotations. It comes from **MimMiner** (van Driel et al., 2006): the full text and clinical synopsis of 5,080 OMIM phenotype records were text-mined for MeSH terms from the *anatomy* (A) and *disease* (C) branches; each record became a feature vector of MeSH-term counts, re-weighted so that common terms count less (an inverse-document-frequency idea) and adjusted using the MeSH hierarchy; two records were compared with the cosine of their vectors. The authors showed that phenotypically similar records tend to involve functionally related genes. Characteristics that matter for us:

* It is **text similarity of clinical descriptions**, so it captures shared *symptoms* (Alzheimer disease type 1 and schizophrenia: 0.410) even when classification differs.
* It covers **every** OMIM disease in the benchmark (no mapping needed).
* It is frozen in 2006: OMIM records written since then are not reflected.

`sem_mondo` captures something different — **where a disease sits in an expert classification** — which is why the two combine well.

---

## 6. Cross-references and mapping between resources

A **cross-reference** (xref) links a term to an identifier in another resource. Treat xrefs with care; they are the glue *and* the main source of error in multi-database projects.

* **Semantics differ.** An xref may mean "exactly the same concept", "broader", "narrower" or merely "related". OBO's bare `xref:` does not say which. MONDO adds qualifiers; the SKOS vocabulary (`skos:exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`, `relatedMatch`) and the **SSSOM** standard (Simple Standard for Sharing Ontological Mappings) make the predicate explicit and add provenance and confidence. MONDO's `property_value: skos:exactMatch OMIM:104300` lines are the SKOS form of its equivalences.
* **Cardinality.** Mappings can be 1:1, 1:many, many:1 or many:many. A MeSH descriptor (Alzheimer Disease, D000544) is broader than dozens of OMIM entries; an OMIM entry may be linked to a specific SCR *and* a general descriptor (our table's OMIM 104300 has `MESH:C536594|MESH:D000544`).
* **Prefix hygiene.** `OMIM:`/`MIM:`, `MESH:`/`MSH:`, `Orphanet:`/`ORPHA:`/`ORDO:`, `UMLS:`/`UMLS_CUI:` all occur in the wild. Normalise before joining.
* **Versioning and obsolescence.** Mappings point to IDs that may be obsolete in the target's current release. Check targets are live; follow `replaced_by` where it exists.
* **Chains multiply errors.** OMIM → MedGen (UMLS CUI) → MeSH → CTD is three hops; each hop can add or lose meaning. Prefer curated direct mappings (MONDO's OMIM equivalences) and record which path produced each link.

The project's disease mapping (`HOW_IT_WORKS.md`, Section 2.3):

```
OMIM ID ──MedGen──────────► name, MeSH IDs
OMIM ID ──MONDO xrefs─────► MONDO term(s) ──► position in the disease DAG (sem_mondo)
                                          └─► MeSH xrefs
OMIM ID / MeSH ID ──CTD MEDIC──► CTD disease ID ──► genes (gene_d)
```

Of the 415 unique OMIM diseases across both benchmarks, 404 have a live MONDO term. The 11 without one are OMIM entries that have since been removed or merged (several have only `obsoleteEquivalent` xrefs on obsolete MONDO classes, e.g. 304900, 601367 "ischemic stroke", 608622 "hypertension, diastolic, resistance to") or that never had a MONDO equivalent.

---

## 7. Semantic similarity: the families

### 7.1 What we are measuring

**Semantic similarity** quantifies how alike two terms are *in meaning*, using the ontology's structure (and optionally annotation statistics) rather than their labels. Pesquita et al. (2009) organise the field along two axes:

* **What is compared.** *Term-level* measures compare two terms. *Entity-level* measures compare two entities (two diseases, two genes), each annotated with a *set* of terms; they either combine term-level scores (pairwise: max, average, best-match average; Section 9) or compare the sets directly (groupwise: simUI, simGIC).
* **How specificity is measured.** *Edge-based* measures count edges; *node-based* (information-content) measures use how informative the shared ancestors are; *hybrid* measures combine graph topology with weights or IC.

### 7.2 Edge-based (path) measures

The simplest idea: the fewer edges between two terms, the more similar they are.

* **Path distance** (Rada et al., 1989): $d(a,b)$ = length of the shortest path between $a$ and $b$ *through a common ancestor*; similarity e.g. $1/(1+d)$. Restricting to paths through a common ancestor matters in a DAG: in the toy DAG, A → D → W → M is a 3-edge path from A to M, but it goes *down* through W, a descendant of both; the meaningful path goes up through R: A → D → N → R → M, length 4.
* **Wu–Palmer** (1994): $\mathrm{sim}_{WP}(a,b) = \dfrac{2\,\mathrm{depth}(\mathrm{LCA})}{\mathrm{depth}(a) + \mathrm{depth}(b)}$, with depth counted in nodes (root = 1). Toy DAG: depth R = 1, N = M = 2, D = 3, A = P = 4, so $\mathrm{sim}_{WP}(A,P) = 6/8 = 0.75$. For W the depth is ambiguous (3 via M, 4 via D), giving 0.857 or 0.75 for (A, W) — a direct consequence of the DAG structure.
* **Leacock–Chodorow** (1998): $\mathrm{sim}_{LC}(a,b) = -\log \dfrac{\mathrm{len}(a,b)}{2 D}$, with path length counted in nodes and $D$ the maximum depth of the ontology.

Their shared weakness is the assumption that **all edges represent the same semantic distance**. They do not: an edge near the root ("disease" → "nervous system disorder") is a huge conceptual step; an edge deep in a well-studied branch ("familial Alzheimer disease" → "Alzheimer disease type 1") is a tiny one. Depth also reflects curation effort: heavily studied areas get deeper hierarchies.

### 7.3 Information-content (node-based) measures

**Information content** quantifies how specific a term is. Let $p(t)$ be the probability that a randomly chosen "thing" is described by $t$ *or any of its descendants*. Then
$$
\mathrm{IC}(t) = -\log p(t).
$$
A term that almost everything falls under ("disease") has $p \approx 1$ and $\mathrm{IC} \approx 0$: knowing that two diseases are both "diseases" tells you nothing. A term that covers only one thing has high IC. Because of the true path rule, $p$ can only grow as you go up, so IC decreases monotonically from leaves to root.

Two ways to estimate $p(t)$:

* **Corpus-based** (Resnik's original): count annotations. $p(t) = \dfrac{\#\text{annotations to } t \text{ or its descendants}}{\#\text{annotations in total}}$. For diseases, one could count how many OMIM entries or how many PubMed articles fall under each class. It reflects usage — and its biases (well-studied diseases get low IC).
* **Intrinsic** (Seco et al., 2004): use only the ontology. $p(t) = \dfrac{|\mathcal{D}(t)|}{N}$, with $|\mathcal{D}(t)|$ the number of descendants including $t$ and $N$ the number of terms. For MONDO ($N = 32{,}109$ live classes): *Alzheimer disease type 1* is a leaf, $\mathrm{IC} = \ln 32109 = 10.38$; *Alzheimer disease* (22 descendants) 7.29; *dementia* (160) 5.30; *neurodegenerative disease* (921) 3.55; *nervous system disorder* (5,505) 1.76; *disease* (31,558) 0.017 (Section 11.6).

The three classic measures built on IC, for terms $a$, $b$ with most informative common ancestor $m = \mathrm{MICA}(a,b)$:

| Measure | Formula | Range | Comments |
|---|---|---|---|
| **Resnik** (1995) | $\mathrm{sim}_R(a,b) = \mathrm{IC}(m)$ | $[0, \infty)$ | depends only on the shared ancestor; $\mathrm{sim}_R(a,a) = \mathrm{IC}(a)$, not 1; two pairs with the same MICA get the same score however far they are from it |
| **Lin** (1998) | $\mathrm{sim}_L(a,b) = \dfrac{2\,\mathrm{IC}(m)}{\mathrm{IC}(a) + \mathrm{IC}(b)}$ | $[0, 1]$ | "shared information over total information"; $\mathrm{sim}_L(a,a) = 1$ |
| **Jiang–Conrath** (1997) | $\mathrm{dist}_{JC}(a,b) = \mathrm{IC}(a) + \mathrm{IC}(b) - 2\,\mathrm{IC}(m)$ | $[0, \infty)$ | a distance; convert with $1/(1+\mathrm{dist})$ |

Lin derived his measure from axioms: similarity should be the information in what two objects have in common divided by the information needed to describe both fully. Note the family resemblance to Tanimoto (Unit D2): "shared over total".

**Toy DAG with intrinsic IC** ($N = 7$, natural logs): $\mathrm{IC}(R) = \ln(7/7) = 0$; $\mathrm{IC}(N) = \ln(7/5) = 0.336$; $\mathrm{IC}(M) = \ln(7/2) = 1.253$; $\mathrm{IC}(D) = \ln(7/4) = 0.560$; $\mathrm{IC}(A) = \mathrm{IC}(P) = \mathrm{IC}(W) = \ln 7 = 1.946$.

* (A, P): common ancestors {D, N, R}; MICA = D. Resnik 0.560; Lin $= 1.119/3.892 = 0.288$; JC distance $3.892 - 1.119 = 2.773$.
* (A, W): common ancestors {D, N, R}; MICA = D. **Exactly the same** Resnik, Lin and JC values as (A, P).
* (W, M): MICA = M; Resnik 1.253; Lin $= 2.506/3.199 = 0.783$.

The (A, P) versus (A, W) tie is instructive: MICA-based measures look only at the single best shared ancestor and ignore that W has *another*, unshared, ancestry (it is also metabolic). Wang's method, next, does see it.

### 7.4 Hybrid measures

Hybrid measures use the whole set of ancestors with weights, or combine IC with topology:

* **Wang et al. (2007)** — weights ancestors by their distance through the graph (Section 8). Uses only the ontology, no annotation corpus.
* **GraSM** (Couto et al.) — averages the IC of *all* "disjunctive" common ancestors instead of only the MICA, addressing multiple inheritance.
* **simGIC** (Pesquita et al.) — a groupwise measure: Jaccard of the two entities' ancestor sets, each term weighted by its IC.
* **simUI** — plain Jaccard of the two ancestor sets (union-intersection).

---

## 8. Wang's method, derived step by step

### 8.1 Intuition

Wang, Du, Payattakool, Yu and Chen (2007) proposed it for the Gene Ontology, and it has become the standard for disease DAGs (MeSH, DO, MONDO) in drug-repositioning and microRNA–disease papers. The idea: the *meaning* of a term is not just the term itself but the term **plus all its ancestors**, each of which contributes part of the meaning. "Alzheimer disease type 1" *means* "a familial Alzheimer disease, which is an Alzheimer disease, which is a dementia, which is …". Ancestors close to the term say a lot about it; distant ones (the root) say very little. Two terms are similar if their meanings — these weighted ancestor sets — overlap heavily.

### 8.2 Definitions

Represent term $A$ by its sub-DAG $\mathrm{DAG}_A = (A, T_A, E_A)$, where $T_A$ is the set of $A$ and all its ancestors, and $E_A$ the edges among them.

**Semantic contribution (S-value)** of each $t \in T_A$ to $A$:
$$
S_A(A) = 1, \qquad
S_A(t) = \max\big\{\, w_e \cdot S_A(t') \;:\; t' \in \mathrm{children}_A(t) \,\big\} \quad (t \neq A),
$$
where $\mathrm{children}_A(t)$ are the children of $t$ *inside* $\mathrm{DAG}_A$, and $w_e \in (0, 1)$ is the **semantic contribution factor** of edge type $e$. Wang et al. used $w = 0.8$ for `is_a` and $0.6$ for `part_of` in GO; disease-DAG work since Wang D. et al. (2010, MeSH, microRNA similarity) commonly uses a single $w = 0.5$, as does this project.

**Semantic value** of $A$: the total meaning,
$$
SV(A) = \sum_{t \in T_A} S_A(t).
$$

**Similarity** of terms $A$ and $B$:
$$
\mathrm{Sim}_W(A, B) = \frac{\displaystyle\sum_{t \in T_A \cap T_B} \big(S_A(t) + S_B(t)\big)}{SV(A) + SV(B)}.
$$

In words: add up, from both sides, the semantic contributions of the shared ancestors, and divide by the total semantics of both terms. It is a weighted "shared over total" — Dice's coefficient on weighted ancestor sets.

### 8.3 Worked example by hand (toy DAG, $w = 0.5$)

**S-values.** Start at the term with 1 and walk upward, multiplying by $w$ at each edge, keeping the *maximum* when a node is reached by several paths.

* $T_A = \{A, D, N, R\}$: $S_A(A) = 1$, $S_A(D) = 0.5$, $S_A(N) = 0.25$, $S_A(R) = 0.125$. $SV(A) = 1.875$. $P$ is identical: $SV(P) = 1.875$.
* $T_W = \{W, D, M, N, R\}$: $S_W(W) = 1$; $S_W(D) = 0.5$ and $S_W(M) = 0.5$ (both direct parents); $S_W(N) = 0.5 \cdot 0.5 = 0.25$; $R$ is reached via N ($0.25 \cdot 0.5 = 0.125$) and via M ($0.5 \cdot 0.5 = 0.25$), so $S_W(R) = \max(0.125, 0.25) = 0.25$. $SV(W) = 1 + 0.5 + 0.5 + 0.25 + 0.25 = 2.5$.
* $T_D = \{D, N, R\}$: 1, 0.5, 0.25; $SV(D) = 1.75$. $T_M = \{M, R\}$: 1, 0.5; $SV(M) = 1.5$. $T_R = \{R\}$: $SV(R) = 1$.

**Similarities.**

| Pair | Shared ancestors and their $S_X + S_Y$ | Numerator | Denominator | Sim |
|---|---|---|---|---|
| A, P | D: 0.5+0.5; N: 0.25+0.25; R: 0.125+0.125 | 1.75 | 3.75 | **0.467** |
| A, W | D: 0.5+0.5; N: 0.25+0.25; R: 0.125+0.25 | 1.875 | 4.375 | **0.429** |
| A, D (child–parent) | D: 0.5+1; N: 0.25+0.5; R: 0.125+0.25 | 2.625 | 3.625 | **0.724** |
| W, M | M: 0.5+1; R: 0.25+0.5 | 2.25 | 4.0 | **0.5625** |
| A, M | R: 0.125+0.5 | 0.625 | 3.375 | **0.185** |
| A, R (term vs root) | R: 0.125+1 | 1.125 | 2.875 | **0.391** |

Observations:

1. **A–P > A–W**, although both pairs share exactly the ancestors {D, N, R}. W carries extra, unshared meaning (its metabolic side, $S_W(M) = 0.5$), which enlarges the denominator. IC-based MICA measures gave a tie (Section 7.3).
2. **Parent–child pairs score highest** (0.724), siblings next (0.467), and pairs joined only at the root lowest (0.185).
3. **Anything is somewhat similar to the root** (A–R = 0.391): the root is in every term's ancestor set. This is harmless when you rank neighbours (everything shares the root), but beware of reading absolute values (Section 10).

### 8.4 Closed form: S-values are powers of $w$

**Claim.** With a single weight $w$ for all edges, $S_A(t) = w^{\,\ell(A,t)}$, where $\ell(A,t)$ is the length of the **shortest** upward path from $A$ to $t$.

**Proof.** Any upward path $A = t_0 \to t_1 \to \cdots \to t_k = t$ yields, by applying the recursion along it, the value $w^k$; the recursion takes the maximum over the last step, so by induction on $k$, $S_A(t) = \max_{\text{paths}} w^{k} = w^{\min k}$, because $w^k$ is decreasing in $k$ for $0 < w < 1$. ∎

So computing S-values is a **shortest-path problem** (breadth-first search from $A$ upward), and with mixed weights ($w_{is\_a} \ne w_{part\_of}$) it is a "most reliable path" problem: maximise a product of weights, equivalently minimise the sum of $-\log w_e$ (Dijkstra's algorithm).

### 8.5 Closed form on a chain: why the root hardly matters

Consider two terms $A$ and $B$ at the same distance $L$ from the root in a tree-like region, whose lowest common ancestor is $k$ steps above each of them (siblings: $k = 1$; cousins: $k = 2$). Then
$$
SV(A) = SV(B) = \sum_{i=0}^{L} w^i = \frac{1 - w^{L+1}}{1 - w},
$$
and the shared ancestors are exactly those at distance $k, k+1, \ldots, L$ from each term, so
$$
\text{numerator} = 2 \sum_{i=k}^{L} w^i = 2 w^k \frac{1 - w^{L-k+1}}{1 - w}.
$$
Therefore
$$
\boxed{\;\mathrm{Sim}_W(A,B) = w^k \, \frac{1 - w^{L-k+1}}{1 - w^{L+1}} \;\xrightarrow{\;L \to \infty\;}\; w^k .\;}
$$
Check on the toy DAG: A and P have $L = 3$, $k = 1$: $0.5 \cdot (1 - 0.5^3)/(1 - 0.5^4) = 0.5 \cdot 0.875 / 0.9375 = 0.467$ ✓.

The formula explains the method's behaviour:

* With $w = 0.5$, **siblings score about 0.5, cousins about 0.25, second cousins about 0.125**: similarity halves with each extra step to the LCA. Depth $L$ barely matters once it is moderate.
* The root's contribution to the numerator is $2w^L$; at depth 7 with $w = 0.5$ that is $2/128 = 0.016$, compared with $2w = 1$ for a shared parent. **A close common ancestor contributes exponentially more than a distant one** (self-check Q1).
* Similarly, for a child $A$ at distance $L$ and its parent: $\mathrm{Sim}_W \to (1+w)/2 = 0.75$ as $L \to \infty$ (toy DAG with $L = 3$: 0.724).
* Larger $w$ (e.g. 0.8) makes distant ancestors matter more: siblings ≈ 0.8, cousins ≈ 0.64, so similarities are higher and flatter.

### 8.6 A real worked example: two Alzheimer diseases in MONDO

Alzheimer disease type 1 (AD1, MONDO:0007088, OMIM 104300) versus Alzheimer disease 2 (AD2, MONDO:0007089, OMIM 104310; the *APOE ε4*-associated late-onset form). Walking upward in MONDO (Section 11.5 prints everything):

```
AD1 (1)                                       AD2 (1)
 ├─ early-onset autosomal dominant AD (0.5)    └─ familial Alzheimer disease (0.5)
 │    ├─ autosomal dominant disease (0.25)          ├─ Alzheimer disease (0.25)
 │    └─ familial Alzheimer disease (0.25)          ├─ hereditary dementia (0.25)
 └─ APP-related brain and vascular                  └─ inherited neurodegenerative disorder (0.25)
    amyloidosis (0.5)
      ├─ cerebrovascular disorder (0.25)
      ├─ hereditary amyloidosis (0.25)
      └─ inherited neurodegenerative disorder (0.25)
 ... 35 terms in total, SV(AD1) = 5.1171875   ... 21 terms in total, SV(AD2) = 3.3359375
```

AD1 has two parents and a rich ancestry (35 terms including itself; minimum distance 7 to *disease*, maximum 10). AD2 has one parent. They share 20 ancestors; the largest contributions to the numerator are (several further ancestors tie at 0.25):

| Shared ancestor | $S_{AD1}$ | $S_{AD2}$ | Sum |
|---|---|---|---|
| familial Alzheimer disease | 0.25 | 0.5 | 0.75 |
| inherited neurodegenerative disorder | 0.25 | 0.25 | 0.5 |
| Alzheimer disease | 0.125 | 0.25 | 0.375 |
| hereditary dementia | 0.125 | 0.25 | 0.375 |
| neurodegenerative disease | 0.125 | 0.125 | 0.25 |
| … 14 more … | | | |
| disease (root) | 0.0078 | 0.0078 | 0.0156 |

Numerator $= 3.859375$, denominator $= 5.1171875 + 3.3359375 = 8.453125$, so
$$
\mathrm{Sim}_W(\mathrm{AD1}, \mathrm{AD2}) = 3.859375 / 8.453125 = 0.4566.
$$
The single closest shared ancestor (*familial Alzheimer disease*) contributes 19% of the numerator; the root contributes 0.4%.

Other project pairs (all $w = 0.5$, Section 11.5), with the MimMiner phenotype similarity for comparison:

| Pair | Wang (`sem_mondo`) | MimMiner (`pheno_mim`) |
|---|---|---|
| AD1 / AD2 | 0.457 | 0.381 |
| type 1 diabetes / type 2 diabetes | 0.512 | 0.395 |
| AD1 / late-onset Parkinson disease | 0.172 | 0.326 |
| AD1 / schizophrenia | 0.145 | 0.410 |
| AD1 / type 2 diabetes | 0.054 | 0.273 |

(The MimMiner values involving AD2 and type 1 diabetes come from Cdataset; the others are identical in both benchmarks.) The ontology ranks the pairs the way a clinician's classification would; MimMiner rates AD1 closer to schizophrenia than to the other Alzheimer form, because their OMIM texts share psychiatric and cognitive vocabulary.

### 8.7 Algorithm, as implemented in the project

```
wang_s_values(term, parents, w):
    S = {term: 1.0}; stack = [term]
    while stack:
        t = stack.pop()
        v = S[t] * w                      # what t can pass to each parent
        for p in parents[t]:
            if v > S.get(p, 0):           # found a better (shorter) path to p
                S[p] = v
                stack.append(p)           # p must re-propagate its new value
    return S
```

This is a **label-correcting** search: a node's value can be improved several times if it is first reached by a long path and later by a shorter one, and each improvement is re-propagated. It terminates because values only increase and are bounded by 1, and there are finitely many paths in a DAG; at termination every node holds the maximum over all paths, which is the recursive definition. A breadth-first search would visit nodes in order of distance and never need corrections; for the small ancestor sets of disease terms (tens of nodes) the difference is irrelevant. The project caches each term's S-values, so building the 313 × 313 Fdataset matrix needs only 313 propagations and about 49,000 cheap dictionary intersections.

### 8.8 Properties of Wang similarity

1. **Symmetric** — the formula is symmetric in $A$ and $B$.
2. **Bounded in [0, 1]** — the numerator sums a subset of the terms of the denominator.
3. **$\mathrm{Sim}_W(A, A) = 1$**, and **$\mathrm{Sim}_W(A, B) < 1$ for $A \neq B$**: if $B \notin T_A$, the term $S_B(B) = 1$ is in the denominator but not the numerator. (If $B$ is an ancestor of $A$, then $A \notin T_B$ in a DAG, and the same argument works with $S_A(A)$.)
4. **Zero iff no common ancestor** — e.g. terms under different MONDO roots. Within one root it is always positive.
5. **Not a metric**, and absolute values depend on $w$ and on the local shape of the DAG (number of parents, depth). Use it to *rank* neighbours.

---

## 9. From terms to entities: max, average, best-match average

A disease may map to **several** terms (several MONDO classes, several MeSH descriptors, or a whole profile of HPO phenotypes). Given term-level similarities $s(a, b)$ between entity $X$'s terms $\{a_1, \ldots, a_m\}$ and $Y$'s terms $\{b_1, \ldots, b_n\}$:

| Aggregation | Formula | Behaviour |
|---|---|---|
| **Maximum** | $\max_{i,j} s(a_i, b_j)$ | optimistic: one strong match suffices |
| **Average** | $\frac{1}{mn}\sum_{i,j} s(a_i, b_j)$ | pessimistic: an entity with diverse terms is not even fully similar to itself |
| **Best-match average (BMA)** | $\dfrac{\sum_i \max_j s(a_i, b_j) + \sum_j \max_i s(a_i, b_j)}{m + n}$ | each term is matched to its best partner, then averaged; Wang et al.'s choice for gene sets |

**Toy example.** Suppose disease $X$ maps to $\{A, W\}$ and disease $Y$ to $\{P\}$. From Section 8.3, $s(A, P) = 0.467$ and $s(W, P) = 0.429$ (W and P relate exactly as W and A do).

* max $= 0.467$
* average $= (0.467 + 0.429)/2 = 0.448$
* BMA $= \big(\max(0.467) + \max(0.429) + \max(0.467, 0.429)\big)/3 = (0.467 + 0.429 + 0.467)/3 = 0.454$

**Which to use?** It depends on what multiple terms *mean*:

* If the terms are **alternative codings of the same disease** (a mapping returned two MONDO classes because of an imperfect cross-reference), only the best-matching coding is meaningful and **max** is right. This is the project's case, and why `wang_similarity` uses max.
* If the terms describe **different aspects** of one entity (a gene's several functions; a disease's many phenotypes), all of them should count and **BMA** is the standard choice. HPO-based disease similarity uses BMA-type aggregation.
* Groupwise alternatives (simGIC, simUI) compare the *union* of the ancestor sets directly and avoid the choice.

---

## 10. Pitfalls

1. **Shallow annotation.** If a disease is mapped to a high-level grouping class (because no specific class exists, or the mapping was lazy), its ancestor set is small and generic; it becomes moderately similar to everything in that branch and highly similar to nothing. Check the distribution of the number of ancestors per mapped term (Exercise 12).
2. **Uneven depth and density.** Heavily studied branches (cancer, neurology) are deeper and bushier than neglected ones. Path-based measures and Wang's S-values are both affected: the same clinical "distance" may be 2 edges in one branch and 5 in another.
3. **Multiple classification axes create cross-system similarity.** MONDO classifies by organ system *and* by genetic mechanism. AD1 and *neurohypophyseal (central) diabetes insipidus* get Wang similarity 0.287 — higher than AD1 versus late-onset Parkinson disease (0.172) — mostly because both are *autosomal dominant diseases* (contribution 0.25 + 0.5) and *hereditary neurological diseases* (0.125 + 0.5) (Section 11.10). Mechanism groupings are biologically real but say little about which *drugs* work. A refinement would be to down-weight or drop mechanism-only grouping classes (MONDO's `disease_grouping` subset and the `MONDO:7770xxx` "disease by …" classes).
4. **The root and high-level classes give a similarity floor.** Within one root, Wang similarity is never 0 (AD1 vs type 2 diabetes: 0.054). For $k$-NN graphs this does not matter (only ranks matter); for anything that uses absolute values (thresholds, averages, attention over raw values), it does.
5. **Obsolete terms.** Mapping files and old datasets point to obsolete classes. An obsolete class has no `is_a` edges, so it would be an isolated node with similarity 0 to everything — or a crash. Filter them (the project does) and follow `replaced_by`/`consider` where possible.
6. **Violations of the true path rule** (a wrong `is_a` edge) propagate into every similarity through that edge. You cannot detect them statistically; they are fixed by reporting them to the ontology's issue tracker.
7. **Imported classes and multiple roots** (Section 4.2): filter by prefix; know which root you are under. In MONDO, OMIM "susceptibility to X" entries live under the separate root *disease susceptibility*, usually as direct children of one grouping class, *inherited disease susceptibility*. In `sem_mondo` this has two effects (Exercise 12): such a disease has similarity **0** to every disease that sits only under the main root — typically including X itself — and similarity about **0.43** to other susceptibility entries placed the same way, whatever the organ (endometriosis susceptibility vs osteoarthritis susceptibility: 0.4286, the sibling formula with $L = 2$). That is an artificial cluster of 33 Fdataset diseases. A fix would link each susceptibility class to the disease it predisposes to, or treat these diseases as unmapped (NaN) so that the model relies on `pheno_mim` for them.
8. **Prefix mismatches** (`MIM:` vs `OMIM:`) silently produce zero mappings.
9. **Version drift.** MONDO is released monthly; mappings and edges change. Record the `data-version` (here `releases/2026-09-01`) with your results.
10. **IC from a biased corpus** (corpus-based IC) rewards obscurity: rare, little-annotated diseases get high IC. Intrinsic IC avoids corpus bias but inherits curation bias in the hierarchy.

---

## 11. Hands-on code (plain Python on the real files, run in order)

Start Python in the **project root** with `"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe"` (or a notebook using that interpreter). Every block was executed in one session in this order; the text under each block is its exact output.

### 11.1 A general OBO parser, run on MONDO

```python
import re, time
from collections import defaultdict, Counter, deque

# value  [{modifiers}]  [! comment]   -- quoted strings may contain anything
LINE = re.compile(r'^(?P<val>"(?:[^"\\]|\\.)*"[^{!]*|[^{!]*?)\s*(?:\{(?P<mod>.*)\})?\s*(?:!\s*(?P<cmt>.*))?$')

def parse_obo(path):
    """Return (header, stanzas). Each stanza is {'_type': 'Term', tag: [(value, modifiers), ...]}."""
    header, stanzas, cur = defaultdict(list), [], None
    for raw in open(path, encoding="utf8"):
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("["):                       # [Term], [Typedef], [Instance]
            cur = defaultdict(list)
            cur["_type"] = line.strip("[]")
            stanzas.append(cur)
            continue
        tag, sep, rest = line.partition(": ")
        if not sep:
            continue
        m = LINE.match(rest)
        val, mod = (m.group("val").strip(), m.group("mod") or "") if m else (rest, "")
        (header if cur is None else cur)[tag].append((val, mod))
    return header, stanzas

t0 = time.time()
mondo_header, mondo_stanzas = parse_obo("data/raw/ontology/mondo.obo")
seconds = time.time() - t0                            # about 5-10 s; not printed, it varies
print("data-version:", mondo_header["data-version"][0][0])
print("stanza types:", Counter(s["_type"] for s in mondo_stanzas))
ids = [s["id"][0][0] for s in mondo_stanzas if s["_type"] == "Term"]
print("Term prefixes (top 6):", Counter(i.split(":")[0] for i in ids).most_common(6))

def first(s, tag, default=""):
    return s[tag][0][0] if tag in s else default

terms = {}                                            # live MONDO classes only
n_obsolete = 0
for s in mondo_stanzas:
    if s["_type"] != "Term" or not first(s, "id").startswith("MONDO:"):
        continue
    if first(s, "is_obsolete") == "true":
        n_obsolete += 1
        continue
    terms[first(s, "id")] = s
parents = {t: [v for v, _ in s.get("is_a", []) if v in terms] for t, s in terms.items()}
children = defaultdict(list)
for c, ps in parents.items():
    for p in ps:
        children[p].append(c)
name = lambda t: first(terms[t], "name")

print("live MONDO classes:", len(terms), "| obsolete MONDO stanzas:", n_obsolete)
print("is_a edges:", sum(map(len, parents.values())),
      "| classes with >1 parent:", sum(len(p) > 1 for p in parents.values()))
print("roots:", [(r, name(r)) for r, p in parents.items() if not p])
```

```text
data-version: releases/2026-09-01
stanza types: Counter({'Term': 63278, 'Typedef': 282})
Term prefixes (top 6): [('MONDO', 36015), ('http', 5788), ('UBERON', 5360), ('HP', 4859), ('GO', 4142), ('NCBITaxon', 2227)]
live MONDO classes: 32109 | obsolete MONDO stanzas: 3906
is_a edges: 46582 | classes with >1 parent: 10722
roots: [('MONDO:0000001', 'disease'), ('MONDO:0021125', 'disease characteristic'), ('MONDO:0021178', 'injury'), ('MONDO:0042489', 'disease susceptibility')]
```

### 11.2 Reading one term

```python
AD1 = "MONDO:0007088"
s = terms[AD1]
print(AD1, "|", name(AD1))
for v, _ in s["synonym"]:
    text, scope = re.match(r'"(.*)" (\w+)', v).groups()
    print(f"   synonym {scope:7s} {text}")
for v, mod in s["xref"]:
    kind = re.findall(r'source="MONDO:(\w+)"', mod)
    print(f"   xref {v:16s} {kind}")
for p in parents[AD1]:
    print("   is_a", p, name(p))
```

```text
MONDO:0007088 | Alzheimer disease type 1
   synonym RELATED AD
   synonym EXACT   AD1
   synonym RELATED Alzheimer disease 1
   synonym EXACT   Alzheimer disease 1, familial
   synonym EXACT   Alzheimer disease, familial, 1
   synonym BROAD   early-onset familial form of Alzheimer disease
   xref DECIPHER:48      ['relatedTo']
   xref DOID:0080348     ['equivalentTo']
   xref GARD:0009465     ['GARD']
   xref MEDGEN:354892    ['equivalentTo', 'MEDGEN']
   xref MESH:C536594     ['equivalentTo']
   xref OMIM:104300      ['equivalentTo']
   xref UMLS:C1863052    ['equivalentTo', 'MEDGEN']
   is_a MONDO:0015140 early-onset autosomal dominant Alzheimer disease
   is_a MONDO:1060190 APP-related brain and vascular amyloidosis
```

### 11.3 Ancestors, depth, LCA

```python
def ancestors(t):
    """All ancestors of t, including t itself."""
    seen, stack = {t}, [t]
    while stack:
        for p in parents[stack.pop()]:
            if p not in seen:
                seen.add(p); stack.append(p)
    return seen

def min_depth(t):                       # BFS upward: shortest path to a root
    dist, q = {t: 0}, deque([t])
    while q:
        x = q.popleft()
        if not parents[x]:
            return dist[x]
        for p in parents[x]:
            if p not in dist:
                dist[p] = dist[x] + 1; q.append(p)

_maxd = {}
def max_depth(t):                       # longest path to a root (memoised DFS)
    if t not in _maxd:
        _maxd[t] = 0 if not parents[t] else 1 + max(max_depth(p) for p in parents[t])
    return _maxd[t]

def lcas(a, b):
    common = ancestors(a) & ancestors(b)
    return {c for c in common if not any(c != o and c in ancestors(o) for o in common)}

terms_of = {"AD1": "MONDO:0007088", "AD2": "MONDO:0007089", "PD": "MONDO:0008199",
            "SCZ": "MONDO:0005090", "T1D": "MONDO:0005147", "T2D": "MONDO:0005148"}
for k, t in terms_of.items():
    print(f"{k:4s} {t} {name(t):32s} ancestors={len(ancestors(t)):3d} "
          f"min_depth={min_depth(t)} max_depth={max_depth(t)}")
for a, b in [("AD1", "AD2"), ("AD1", "PD"), ("AD1", "SCZ"), ("T1D", "T2D")]:
    print(f"LCA({a},{b}) =", sorted(name(c) for c in lcas(terms_of[a], terms_of[b])))
```

```text
AD1  MONDO:0007088 Alzheimer disease type 1         ancestors= 35 min_depth=7 max_depth=10
AD2  MONDO:0007089 Alzheimer disease 2              ancestors= 21 min_depth=7 max_depth=9
PD   MONDO:0008199 late-onset Parkinson disease     ancestors= 14 min_depth=6 max_depth=9
SCZ  MONDO:0005090 schizophrenia                    ancestors= 12 min_depth=6 max_depth=8
T1D  MONDO:0005147 type 1 diabetes mellitus         ancestors= 16 min_depth=5 max_depth=7
T2D  MONDO:0005148 type 2 diabetes mellitus         ancestors= 12 min_depth=6 max_depth=7
LCA(AD1,AD2) = ['familial Alzheimer disease']
LCA(AD1,PD) = ['brain disorder', 'hereditary neurological disease']
LCA(AD1,SCZ) = ['brain disorder', 'cognitive disorder']
LCA(T1D,T2D) = ['diabetes mellitus']
```

### 11.4 The toy DAG: Wang by hand, checked with the project's functions

```python
import sys
import numpy as np
sys.path.insert(0, "src")
from drepo.similarity import wang_s_values, wang_similarity

toy = {"A": ["D"], "P": ["D"], "W": ["D", "M"], "D": ["N"], "N": ["R"], "M": ["R"], "R": []}
for t in ["A", "W", "D", "M", "R"]:
    S = wang_s_values(t, toy, w=0.5)
    print(f"S_{t} =", S, " SV =", sum(S.values()))

order = ["A", "P", "W", "D", "M", "R"]
Sim = wang_similarity([[t] for t in order], toy, w=0.5)
print("      " + "".join(f"{t:>7s}" for t in order))
for t, row in zip(order, Sim):
    print(f"{t:>6s}" + "".join(f"{v:7.3f}" for v in row))
```

```text
S_A = {'A': 1.0, 'D': 0.5, 'N': 0.25, 'R': 0.125}  SV = 1.875
S_W = {'W': 1.0, 'D': 0.5, 'M': 0.5, 'R': 0.25, 'N': 0.25}  SV = 2.5
S_D = {'D': 1.0, 'N': 0.5, 'R': 0.25}  SV = 1.75
S_M = {'M': 1.0, 'R': 0.5}  SV = 1.5
S_R = {'R': 1.0}  SV = 1.0
            A      P      W      D      M      R
     A  1.000  0.467  0.429  0.724  0.185  0.391
     P  0.467  1.000  0.429  0.724  0.185  0.391
     W  0.429  0.429  1.000  0.647  0.562  0.357
     D  0.724  0.724  0.647  1.000  0.231  0.455
     M  0.185  0.185  0.562  0.231  1.000  0.600
     R  0.391  0.391  0.357  0.455  0.600  1.000
```

### 11.5 Wang similarity on MONDO: Alzheimer disease type 1

```python
S_AD1 = wang_s_values(terms_of["AD1"], parents)
print(len(S_AD1), "terms in T_AD1, SV =", sum(S_AD1.values()))
for t, v in sorted(S_AD1.items(), key=lambda kv: (-kv[1], name(kv[0])))[:9]:
    print(f"   {v:.4f}  {t}  {name(t)}")
print("   ...  root:", S_AD1["MONDO:0000001"])

S_AD2 = wang_s_values(terms_of["AD2"], parents)
common = S_AD1.keys() & S_AD2.keys()
num = sum(S_AD1[t] + S_AD2[t] for t in common)
den = sum(S_AD1.values()) + sum(S_AD2.values())
print(f"AD1 vs AD2: {len(common)} shared ancestors, numerator {num}, denominator {den}, sim {num/den:.4f}")
for t in sorted(common, key=lambda t: (-(S_AD1[t] + S_AD2[t]), name(t)))[:5]:
    print(f"   {S_AD1[t]:.4f} + {S_AD2[t]:.4f} = {S_AD1[t] + S_AD2[t]:.4f}  {name(t)}")

pairs = [("AD1", "AD2"), ("T1D", "T2D"), ("AD1", "PD"), ("AD1", "SCZ"), ("AD1", "T2D")]
for a, b in pairs:
    s = wang_similarity([[terms_of[a]], [terms_of[b]]], parents)[0, 1]
    print(f"Wang({a},{b}) = {s:.4f}")
```

```text
35 terms in T_AD1, SV = 5.1171875
   1.0000  MONDO:0007088  Alzheimer disease type 1
   0.5000  MONDO:1060190  APP-related brain and vascular amyloidosis
   0.5000  MONDO:0015140  early-onset autosomal dominant Alzheimer disease
   0.2500  MONDO:0000426  autosomal dominant disease
   0.2500  MONDO:0011057  cerebrovascular disorder
   0.2500  MONDO:0100087  familial Alzheimer disease
   0.2500  MONDO:0018634  hereditary amyloidosis
   0.2500  MONDO:0024237  inherited neurodegenerative disorder
   0.1250  MONDO:0004975  Alzheimer disease
   ...  root: 0.0078125
AD1 vs AD2: 20 shared ancestors, numerator 3.859375, denominator 8.453125, sim 0.4566
   0.2500 + 0.5000 = 0.7500  familial Alzheimer disease
   0.2500 + 0.2500 = 0.5000  inherited neurodegenerative disorder
   0.1250 + 0.2500 = 0.3750  Alzheimer disease
   0.1250 + 0.2500 = 0.3750  hereditary dementia
   0.1250 + 0.1250 = 0.2500  hereditary neurological disease
Wang(AD1,AD2) = 0.4566
Wang(T1D,T2D) = 0.5124
Wang(AD1,PD) = 0.1715
Wang(AD1,SCZ) = 0.1452
Wang(AD1,T2D) = 0.0542
```

### 11.6 Information content: Resnik, Lin, Jiang–Conrath on MONDO

```python
import math
n_desc = Counter()
for t in terms:                       # every class counts once for each of its ancestors
    for a in ancestors(t):
        n_desc[a] += 1
N = len(terms)
IC = lambda t: -math.log(n_desc[t] / N)
for t in ["MONDO:0007088", "MONDO:0004975", "MONDO:0001627", "MONDO:0005559",
          "MONDO:0005071", "MONDO:0000001"]:
    print(f"   IC={IC(t):6.3f}  descendants={n_desc[t]:5d}  {name(t)}")

def ic_measures(a, b):
    common = ancestors(a) & ancestors(b)
    mica = max(common, key=IC)
    res = IC(mica)
    return name(mica), res, 2 * res / (IC(a) + IC(b)), IC(a) + IC(b) - 2 * res

print(f"{'pair':9s} {'MICA':28s} {'Resnik':>7s} {'Lin':>6s} {'JC dist':>8s}")
for a, b in pairs[:4]:
    mica, r, l, jc = ic_measures(terms_of[a], terms_of[b])
    print(f"{a + '/' + b:9s} {mica:28s} {r:7.3f} {l:6.3f} {jc:8.3f}")
```

```text
   IC=10.377  descendants=    1  Alzheimer disease type 1
   IC= 7.286  descendants=   22  Alzheimer disease
   IC= 5.302  descendants=  160  dementia
   IC= 3.551  descendants=  921  neurodegenerative disease
   IC= 1.763  descendants= 5505  nervous system disorder
   IC= 0.017  descendants=31558  disease
pair      MICA                          Resnik    Lin  JC dist
AD1/AD2   familial Alzheimer disease     7.544  0.727    5.666
T1D/T2D   diabetes mellitus              6.485  0.716    5.145
AD1/PD    brain disorder                 3.121  0.334   12.432
AD1/SCZ   cognitive disorder             5.069  0.569    7.672
```

### 11.7 Term sets: max, average and best-match average

```python
def set_similarity(X, Y, par, how="max", w=0.5):
    terms_xy = list(X) + list(Y)
    M = wang_similarity([[t] for t in terms_xy], par, w)[:len(X), len(X):]   # |X| x |Y| block
    if how == "max":
        return M.max()
    if how == "avg":
        return M.mean()
    if how == "bma":
        return (M.max(1).sum() + M.max(0).sum()) / (len(X) + len(Y))

for how in ["max", "avg", "bma"]:
    print(how, round(set_similarity(["A", "W"], ["P"], toy, how), 4))
```

```text
max 0.4667
avg 0.4476
bma 0.454
```

### 11.8 Cross-references: OMIM → MONDO, OMIM → DO, and the prefix trap

```python
omim2mondo = defaultdict(set)
qualifiers = Counter()
for t, s in terms.items():
    for v, mod in s.get("xref", []):
        if v.startswith("OMIM:"):
            omim2mondo[v[5:]].add(t)
            qualifiers.update(re.findall(r'source="MONDO:(\w+)"', mod))
print("OMIM ids on live MONDO classes:", len(omim2mondo),
      "| mapping to >1 class:", sum(len(v) > 1 for v in omim2mondo.values()))
print("qualifiers on those xrefs:", dict(qualifiers))

doid_header, doid_stanzas = parse_obo("data/raw/ontology/doid.obo")
doid_live = [s for s in doid_stanzas if s["_type"] == "Term" and first(s, "id").startswith("DOID:")
             and first(s, "is_obsolete") != "true"]
xref_prefixes = Counter(v.split(":")[0] for s in doid_live for v, _ in s.get("xref", []))
print("DOID", doid_header["data-version"][0][0], "| live terms:", len(doid_live),
      "| OMIM-like xref prefixes:", {k: xref_prefixes[k] for k in ["OMIM", "MIM"]})

import pandas as pd, scipy.io as sio
diseases = pd.read_csv("data/interim/diseases_all.csv", dtype=str).fillna("")
omim2doid = {v[4:] for s in doid_live for v, _ in s.get("xref", []) if v.startswith("MIM:")}
for bench in ["Fdataset", "Cdataset"]:
    mat = sio.loadmat(f"data/raw/benchmarks/{bench}.mat")
    omims = [str(x[0])[1:] for x in mat["Wdname"].ravel()]          # 'D104300' -> '104300'
    in_mondo = np.mean([o in omim2mondo for o in omims])
    in_doid = np.mean([o in omim2doid for o in omims])
    print(f"{bench}: {len(omims)} diseases, MONDO coverage {in_mondo:.1%}, DO coverage {in_doid:.1%}")

unmapped = diseases[diseases.mondo == ""]
print("unmapped:", ", ".join(unmapped.omim))
obs = [s for s in mondo_stanzas if s["_type"] == "Term" and first(s, "is_obsolete") == "true"
       and any(v == "OMIM:304900" for v, _ in s.get("xref", []))][0]
print("OMIM 304900 ->", first(obs, "id"), "|", first(obs, "name"))
```

```text
OMIM ids on live MONDO classes: 9904 | mapping to >1 class: 0
qualifiers on those xrefs: {'equivalentTo': 9773, 'equivalentObsolete': 131}
DOID releases/2026-09-30/doid.obo | live terms: 12335 | OMIM-like xref prefixes: {'OMIM': 0, 'MIM': 6560}
Fdataset: 313 diseases, MONDO coverage 97.1%, DO coverage 59.1%
Cdataset: 409 diseases, MONDO coverage 97.3%, DO coverage 64.5%
unmapped: 106400, 133600, 144400, 146850, 212110, 304900, 305300, 601367, 602439, 605839, 608622
OMIM 304900 -> MONDO:0010582 | obsolete diabetes insipidus, neurohypophyseal type, X-linked inheritance
```

### 11.9 MeSH tree numbers (from the CTD copy of MeSH in the project)

```python
cols = ["DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition", "ParentIDs",
        "TreeNumbers", "ParentTreeNumbers", "Synonyms", "SlimMappings"]
medic = pd.read_csv("data/raw/ctd/CTD_diseases.tsv.gz", sep="\t", comment="#",
                    names=cols, dtype=str).fillna("")
tree2name = {tn: r.DiseaseName for r in medic.itertuples() for tn in r.TreeNumbers.split("|") if tn}
ad = medic[medic.DiseaseID == "MESH:D000544"].iloc[0]
print(ad.DiseaseName, ad.DiseaseID, "tree numbers:", ad.TreeNumbers.split("|"))
for tn in ad.TreeNumbers.split("|"):
    parts = tn.split(".")
    chain = [".".join(parts[:k]) for k in range(len(parts), 0, -1)]   # truncate = go up
    print("  " + "  ->  ".join(f"{c} {tree2name.get(c, '?')}" for c in chain))
```

```text
Alzheimer Disease MESH:D000544 tree numbers: ['C10.228.140.380.100', 'C10.574.945.249', 'F03.615.400.100']
  C10.228.140.380.100 Alzheimer Disease  ->  C10.228.140.380 Dementia  ->  C10.228.140 Brain Diseases  ->  C10.228 Central Nervous System Diseases  ->  C10 Nervous System Diseases
  C10.574.945.249 Alzheimer Disease  ->  C10.574.945 Tauopathies  ->  C10.574 Neurodegenerative Diseases  ->  C10 Nervous System Diseases
  F03.615.400.100 Alzheimer Disease  ->  F03.615.400 Dementia  ->  F03.615 Neurocognitive Disorders  ->  F03 Mental Disorders
```

### 11.10 Rebuilding `sem_mondo` for Fdataset and comparing it with `pheno_mim`

```python
from scipy.stats import spearmanr
mat = sio.loadmat("data/raw/benchmarks/Fdataset.mat")
omims = [str(x[0])[1:] for x in mat["Wdname"].ravel()]
ds = diseases.set_index("omim").loc[omims]
S_sem = wang_similarity([m.split("|") if m else [] for m in ds.mondo], parents)   # as in main()
S_mim = mat["disease"].astype(float)
print("sem_mondo", S_sem.shape, "| diseases without MONDO term:", int((ds.mondo == "").sum()))

iu = np.triu_indices(len(omims), 1)
ok = ~np.isnan(S_sem[iu])
print(f"mean Wang {S_sem[iu][ok].mean():.3f}  mean MimMiner {S_mim[iu][ok].mean():.3f}  "
      f"Spearman {spearmanr(S_sem[iu][ok], S_mim[iu][ok])[0]:.3f}")

i = omims.index("104300")
for label, S in [("sem_mondo", S_sem), ("pheno_mim", S_mim)]:
    row = np.nan_to_num(S[i].copy(), nan=-1.0); row[i] = -1
    print(label, [(ds.name.iloc[j][:30], round(float(S[i, j]), 3)) for j in np.argsort(-row)[:5]])

# why is AD1 similar to central diabetes insipidus?
DI = ds.mondo.iloc[omims.index("125700")]
S_DI = wang_s_values(DI, parents)
print("125700 ->", DI, name(DI))
for t in sorted(S_AD1.keys() & S_DI.keys(), key=lambda t: (-(S_AD1[t] + S_DI[t]), name(t)))[:4]:
    print(f"   {S_AD1[t]:.4f} + {S_DI[t]:.4f}  {name(t)}")
```

```text
sem_mondo (313, 313) | diseases without MONDO term: 9
mean Wang 0.127  mean MimMiner 0.097  Spearman 0.050
sem_mondo [('Alzheimer disease, familial ea', 0.544), ('Lewy body dementia', 0.298), ('Central diabetes insipidus', 0.287), ('Ataxia, early-onset, with ocul', 0.279), ('Cerebral arteriopathy, autosom', 0.279)]
pheno_mim [('Huntington disease', 0.46), ('Alzheimer disease, familial ea', 0.419), ('Schizophrenia', 0.41), ('Ischemic stroke', 0.37), ('Unverricht-Lundborg syndrome', 0.363)]
125700 -> MONDO:0007450 neurohypophyseal diabetes insipidus
   0.2500 + 0.5000  autosomal dominant disease
   0.1250 + 0.5000  hereditary neurological disease
   0.1250 + 0.2500  autosomal genetic disease
   0.1250 + 0.2500  brain disorder
```

---

## 12. In this project: the code, line by line

### 12.1 `scripts/02_build_features.py::parse_mondo`

```python
# quoted from scripts/02_build_features.py, parse_mondo
def parse_mondo():
    terms, cur = {}, None
    for line in open(RAW / "ontology" / "mondo.obo", encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("["):
            cur = None
            if line == "[Term]":
                cur = {"id": None, "name": "", "is_a": [], "xrefs": [], "obsolete": False}
        elif cur is not None:
            if line.startswith("id: "):
                cur["id"] = line[4:]
                terms[cur["id"]] = cur
            elif line.startswith("name: "):
                cur["name"] = line[6:]
            elif line.startswith("is_a: "):
                cur["is_a"].append(line[6:].split(" ")[0])
            elif line.startswith("xref: "):
                cur["xrefs"].append(line[6:].split(" ")[0])
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
    terms = {k: v for k, v in terms.items() if k.startswith("MONDO:") and not v["obsolete"]}
    parents = {k: [p for p in v["is_a"] if p in terms] for k, v in terms.items()}
    omim2mondo, mondo2mesh = defaultdict(list), defaultdict(list)
    for k, v in terms.items():
        for x in v["xrefs"]:
            if x.startswith("OMIM:"):
                omim2mondo[x[5:]].append(k)
            elif x.startswith("MESH:"):
                mondo2mesh[k].append(x)
    return terms, parents, omim2mondo, mondo2mesh
```

Line by line:

* **Stanza detection.** Any line starting with `[` ends the previous stanza. Only `[Term]` opens a new record (`cur`); `[Typedef]` sets `cur = None`, so relation definitions are skipped.
* **Header lines** before the first stanza are skipped automatically because `cur` starts as `None`.
* **`id:`** registers the record in `terms` immediately (the dict is shared, so later tags fill it in).
* **`is_a:` and `xref:`** keep only the first whitespace-separated token: `MONDO:0015140 {source=…} ! early-onset …` → `MONDO:0015140`, discarding modifiers and the comment. That is safe because IDs contain no spaces. It also means mapping qualifiers are *not* read: an `OMIM:` xref counts whatever its qualifier (in the current release they are all equivalence-type, Section 11.8).
* **`is_obsolete: true`** marks retired classes.
* **Filtering**: keep `MONDO:` IDs only (drops the imported UBERON, HP, GO, CHEBI… stanzas) and drop obsolete classes. Then `parents` keeps only parent IDs that survived the filter, so the DAG never points to a missing node.
* **Indexes**: `omim2mondo` (OMIM number → list of MONDO classes, a *list* because nothing in the format forbids several) and `mondo2mesh` (MONDO class → MeSH xrefs, used to reach CTD).
* Not parsed: synonyms, definitions, `relationship:` edges (`part_of`, `disease_has_location`, …). Wang similarity here uses only `is_a` with one weight.

### 12.2 `scripts/02_build_features.py::disease_table`

```python
# quoted from scripts/02_build_features.py, disease_table (abridged)
terms, parents, omim2mondo, mondo2mesh = parse_mondo()
...
for o in omims:
    mondo = omim2mondo.get(o, [])
    mesh = set()
    for m in mondo:
        mesh.update(mondo2mesh.get(m, []))
    ...
    rows.append({"omim": o, "name": name, "mondo": "|".join(mondo), "ctd_ids": "|".join(sorted(ctd))})
return pd.DataFrame(rows), parents
```

For each benchmark OMIM number, it records the MONDO class(es) as a `|`-joined string (empty if none), collects MeSH IDs through MONDO and MedGen, and returns the `parents` dict for the similarity step. The readable result is `data/interim/diseases_all.csv` (`omim, name, mondo, ctd_ids, n_genes`).

### 12.3 `src/drepo/similarity.py::wang_s_values` and `wang_similarity`

```python
# quoted from src/drepo/similarity.py
def wang_s_values(term, parents, w=0.5):
    S = {term: 1.0}
    stack = [term]
    while stack:
        t = stack.pop()
        v = S[t] * w
        for p in parents.get(t, ()):
            if v > S.get(p, 0.0):
                S[p] = v
                stack.append(p)
    return S

def wang_similarity(term_sets, parents, w=0.5):
    cache = {}
    def sv(t):
        if t not in cache:
            cache[t] = wang_s_values(t, parents, w)
        return cache[t]
    def term_sim(a, b):
        if a == b:
            return 1.0
        Sa, Sb = sv(a), sv(b)
        common = Sa.keys() & Sb.keys()
        if not common:
            return 0.0
        num = sum(Sa[t] + Sb[t] for t in common)
        return num / (sum(Sa.values()) + sum(Sb.values()))
    n = len(term_sets)
    S = np.full((n, n), np.nan)
    for i in range(n):
        if not term_sets[i]:
            continue
        for j in range(i, n):
            if not term_sets[j]:
                continue
            s = max(term_sim(a, b) for a in term_sets[i] for b in term_sets[j])
            S[i, j] = S[j, i] = s
    np.fill_diagonal(S, 1.0)
    return S
```

* `wang_s_values` is the label-correcting propagation of Section 8.7: `S` holds the best value found so far; whenever a parent's value improves, the parent is pushed back on the stack so its own parents get updated. `parents.get(t, ())` makes unknown or root terms safe. The result is $\{t: S_A(t)\}$ for $t \in T_A$; by Section 8.4 each value is $0.5^{\text{shortest distance}}$.
* `wang_similarity` takes one list of terms per entity (disease), aligned with the benchmark order.
  * `cache` memoises S-values per term, so each term is propagated once.
  * `term_sim` implements the formula of Section 8.2 directly: intersection of the two key sets, sum of both sides' contributions, divided by the two semantic values. Identical terms short-circuit to 1; no shared ancestor (different roots) gives 0.
  * Entities with an empty term list are skipped, leaving `NaN` rows and columns — the "no invented similarity" rule.
  * Only the upper triangle is computed (`j` from `i`), mirrored to keep $S$ symmetric.
  * **Several terms per entity → maximum** over all term pairs (Section 9 and self-check Q2).
  * The diagonal is set to 1 for every entity, including unmapped ones; the coverage check in `main()` treats rows whose only value is the diagonal as uncovered.

The call in `main()`:

```python
# quoted from scripts/02_build_features.py, main
ds = dis_all.set_index("omim").loc[omims]          # benchmark disease order
S_sem = sim.wang_similarity([m.split("|") if m else [] for m in ds.mondo], mondo_parents)
views_d = {"pheno_mim": S_mim, "sem_mondo": S_sem, "gene_d": S_gd}
```

`m.split("|") if m else []` turns the stored string back into a list (empty string → empty list → NaN row). `w` keeps its default 0.5. Section 11.10 reproduces this matrix exactly.

---

## 13. Common mistakes and misconceptions

1. **Joining on labels.** Labels change and are ambiguous; join on IDs (and normalised prefixes).
2. **Treating the hierarchy as a tree.** A term can have several parents, several depths and several LCAs. Code that follows "the" parent silently drops ancestry.
3. **Forgetting to filter imported and obsolete terms** in MONDO's OBO file.
4. **Ignoring prefix variants** (`MIM:` vs `OMIM:`): zero matches, no error.
5. **Assuming xrefs mean equivalence.** Check qualifiers (MONDO) or mapping predicates (SSSOM); MeSH descriptors are often broader than OMIM entries.
6. **Reading absolute Wang values as probabilities.** They depend on $w$ and DAG shape, and have a floor above 0 within a root. Use ranks.
7. **"Closer to the root means more similar."** It is the opposite: what matters is how close the *shared* ancestor is to the two terms.
8. **Using Resnik as if it were normalised.** Resnik is in nats/bits, not [0, 1]; self-similarity equals the term's IC.
9. **Mixing corpus-based IC from one annotation set with another.** IC is only meaningful relative to the corpus that defined it.
10. **Using average aggregation for alternative codings.** It punishes entities that happen to have several equivalent codes.
11. **Not recording the ontology version.** MONDO changes monthly; results are not reproducible without `data-version`.
12. **Assuming ontology similarity equals clinical/therapeutic similarity.** Mechanism groupings (autosomal dominant disease) link diseases that no drug links.

---

## 14. Exercises

Difficulty: ★ conceptual, ★★ calculation / derivation, ★★★ coding or open-ended. Code solutions assume Section 11 has been run in the same session.

**Exercise 1 (★).** Using the stanza in Section 2.2, answer: (a) What is the term's ID and what would its full IRI be? (b) Which synonyms could safely be used to recognise this disease in text? (c) Which xrefs are equivalences? (d) Why does it have two `is_a` lines, and what does the text after `!` mean to a parser?

<details><summary>Solution</summary>

(a) `MONDO:0007088`; IRI `http://purl.obolibrary.org/obo/MONDO_0007088`.
(b) The `EXACT` ones: "AD1" (an abbreviation, ambiguous in free text) and "Alzheimer disease, familial, 1". The `BROAD` synonym "early-onset familial form of Alzheimer disease" covers more than this specific *APP*-linked type, so matching it would over-annotate; the RELATED "AD" is far too ambiguous.
(c) Those with `source="MONDO:equivalentTo"`: DOID:0080348, MESH:C536594, OMIM:104300, MEDGEN:354892 and UMLS:C1863052 (DECIPHER:48 is `relatedTo`; GARD has only a source tag).
(d) Multiple inheritance: AD1 is a kind of early-onset autosomal dominant Alzheimer disease *and* a kind of APP-related brain and vascular amyloidosis (classification by clinical form and by molecular mechanism). Text after `!` is a human-readable comment (the parent's label); parsers ignore it.
</details>

**Exercise 2 (★).** In the toy DAG (Section 3.4), list the common ancestors, the LCAs and the shortest path through a common ancestor for (P, W), (D, M) and (W, R). Compute Wu–Palmer for (P, W) with both possible depths of W.

<details><summary>Solution</summary>

(P, W): $\mathcal{A}(P) = \{P, D, N, R\}$, $\mathcal{A}(W) = \{W, D, M, N, R\}$; common $\{D, N, R\}$; LCA $\{D\}$; path P–D–W, length 2. (D, M): common $\{R\}$; LCA $\{R\}$; path D–N–R–M, length 3. (W, R): R is an ancestor of W, so common $= \{R\}$, LCA $= \{R\}$, path length = distance from W to R = 2 (via M).
Wu–Palmer (node depth, root = 1): depth(D) = 3, depth(P) = 4; depth(W) = 3 (via M) gives $6/7 = 0.857$; depth(W) = 4 (via D) gives $6/8 = 0.75$. The ambiguity is inherent to DAGs; implementations must pick a convention (often the maximum depth).
</details>

**Exercise 3 (★★).** Recompute Wang similarity for (A, W) and (A, P) in the toy DAG with $w = 0.8$. How does the gap between them change compared with $w = 0.5$?

<details><summary>Solution</summary>

$S_A$: A 1, D 0.8, N 0.64, R 0.512; $SV(A) = 2.952$ (same for P).
$S_W$: W 1, D 0.8, M 0.8, N 0.64, R $= \max(0.64 \cdot 0.8, 0.8 \cdot 0.8) = 0.64$; $SV(W) = 3.88$.
(A, P): numerator $2(0.8 + 0.64 + 0.512) = 3.904$; denominator $5.904$; Sim $= 0.661$.
(A, W): numerator $(0.8 + 0.8) + (0.64 + 0.64) + (0.512 + 0.64) = 4.032$; denominator $6.832$; Sim $= 0.590$.

```python
for w in [0.5, 0.8]:
    Sw = wang_similarity([["A"], ["P"], ["W"]], toy, w=w)
    print(f"w={w}: A-P {Sw[0,1]:.4f}  A-W {Sw[0,2]:.4f}  gap {Sw[0,1]-Sw[0,2]:.4f}")
```

```text
w=0.5: A-P 0.4667  A-W 0.4286  gap 0.0381
w=0.8: A-P 0.6612  A-W 0.5902  gap 0.0711
```

Both similarities rise with $w$ (distant ancestors weigh more) and the gap grows from 0.038 to 0.071 here, because with a larger $w$ W's extra branch (M and the better path to R) adds more unshared semantic value.
</details>

**Exercise 4 (★★).** (a) Using the closed form of Section 8.4, show that a term $A$ at distance $L$ from the root along a chain, and its parent, have $\mathrm{Sim}_W = (1+w)(1-w^{L})/\big((1-w^{L+1}) + (1-w^{L})\big)$ and that this tends to $(1+w)/2$. (b) Evaluate for the toy pair (A, D) ($L = 3$, $w = 0.5$).

<details><summary>Solution</summary>

(a) Write $G_n = \sum_{i=0}^{n-1} w^i = (1-w^n)/(1-w)$. $SV(A) = G_{L+1}$, $SV(\mathrm{parent}) = G_L$. Every ancestor of the parent is shared. For a shared term at distance $i$ from the parent ($i = 0, \ldots, L-1$), $S_{\mathrm{parent}} = w^i$ and $S_A = w^{i+1}$, so the numerator is $\sum_{i=0}^{L-1}(w^i + w^{i+1}) = (1+w)G_L$. Hence
$\mathrm{Sim}_W = \dfrac{(1+w) G_L}{G_{L+1} + G_L} = \dfrac{(1+w)(1-w^L)}{(1-w^{L+1}) + (1-w^L)}$. As $L \to \infty$, $w^L \to 0$, giving $(1+w)/2$.
(b) $(1.5)(1 - 0.125)/\big((1 - 0.0625) + (1 - 0.125)\big) = 1.3125/1.8125 = 0.7241$, matching Section 8.3. With $w = 0.5$, a parent–child pair tends to 0.75, siblings to 0.5, cousins to 0.25.
</details>

**Exercise 5 (★★).** Derive a formula for $\mathrm{Sim}_W(A, \mathrm{root})$ and compute it for AD1 in MONDO. What does it tell you about interpreting absolute values?

<details><summary>Solution</summary>

$T_{\mathrm{root}} = \{\mathrm{root}\}$, $SV(\mathrm{root}) = 1$, shared set $= \{\mathrm{root}\}$, so $\mathrm{Sim}_W(A, \mathrm{root}) = \dfrac{S_A(\mathrm{root}) + 1}{SV(A) + 1} = \dfrac{w^{d_{\min}} + 1}{SV(A) + 1}$. For AD1: $d_{\min} = 7$, $S = 0.0078125$, $SV = 5.1171875$: $(1.0078125)/(6.1171875) = 0.1648$.

```python
print(round(wang_similarity([[terms_of["AD1"]], ["MONDO:0000001"]], parents)[0, 1], 4))
```

```text
0.1648
```

"AD1 is 16% similar to *disease*" is meaningless clinically, and it is *higher* than AD1's similarity to type 2 diabetes (0.054). Absolute Wang values mix specificity of the shared part with the size of each term's ancestry; compare values only as ranks within one term's row, and never threshold them without looking at the distribution.
</details>

**Exercise 6 (★★).** Toy DAG, intrinsic IC. (a) Compute Resnik, Lin and Jiang–Conrath for (A, P), (A, W), (W, M), (A, M). (b) Which pairs can IC measures not distinguish, and why? (c) Is $\mathrm{sim}_{Lin}(A, R)$ defined?

<details><summary>Solution</summary>

IC values from Section 7.3: R 0, N 0.336, M 1.253, D 0.560, A = P = W 1.946.
(a) (A, P) and (A, W): MICA D; Resnik 0.560, Lin $1.119/3.892 = 0.288$, JC $2.773$. (W, M): MICA M; Resnik 1.253, Lin $2.506/(1.946 + 1.253) = 0.783$, JC $0.693$. (A, M): MICA R; Resnik 0, Lin 0, JC $3.199$.
(b) (A, P) vs (A, W): identical, because these measures use only the MICA, and both pairs have MICA D; W's additional, unshared parent M is invisible. Wang (0.467 vs 0.429) and groupwise measures (simGIC, simUI) do see it.
(c) $\mathrm{sim}_{Lin}(A, R) = 2 \cdot 0/(1.946 + 0) = 0$ — defined, but note that $\mathrm{sim}_{Lin}(R, R) = 0/0$ is undefined; implementations special-case it.

```python
def toy_anc(t):
    out, st = {t}, [t]
    while st:
        for p in toy[st.pop()]:
            if p not in out: out.add(p); st.append(p)
    return out
toy_desc = Counter(a for t in toy for a in toy_anc(t))
tIC = {t: math.log(len(toy) / toy_desc[t]) for t in toy}
for a, b in [("A", "P"), ("A", "W"), ("W", "M"), ("A", "M")]:
    mica = max(toy_anc(a) & toy_anc(b), key=tIC.get)
    r = tIC[mica]
    print(f"{a}-{b}: MICA {mica}  Resnik {r:.3f}  Lin {2*r/(tIC[a]+tIC[b]):.3f}  JC {tIC[a]+tIC[b]-2*r:.3f}")
```

```text
A-P: MICA D  Resnik 0.560  Lin 0.288  JC 2.773
A-W: MICA D  Resnik 0.560  Lin 0.288  JC 2.773
W-M: MICA M  Resnik 1.253  Lin 0.783  JC 0.693
A-M: MICA R  Resnik 0.000  Lin 0.000  JC 3.199
```
</details>

**Exercise 7 (★★★).** Suppose disease X were mapped to two MONDO classes, {AD1, late-onset Parkinson disease}, and disease Y to {AD2}. Compute the max, average and BMA similarities with `set_similarity`, and argue which is appropriate if (a) X's two classes are alternative codings produced by a sloppy mapping, (b) X is a genuine overlap syndrome with features of both diseases.

<details><summary>Solution</summary>

```python
X, Y = [terms_of["AD1"], terms_of["PD"]], [terms_of["AD2"]]
for how in ["max", "avg", "bma"]:
    print(how, round(set_similarity(X, Y, parents, how), 4))
```

```text
max 0.4566
avg 0.333
bma 0.3742
```

(a) If only one of the codings is right (and one cannot tell which), **max** reports how similar Y is to the best interpretation of X — the project's choice. It is robust to a spurious extra code (adding a wrong code can only increase similarity to *some* diseases, and only if that code is close to them), which is a known risk: a junk code close to everything inflates everything.
(b) If both classes describe real aspects of X, **BMA** credits Y for matching the Alzheimer aspect well while recording that the Parkinson aspect is matched poorly; average does the same more harshly. BMA is the usual choice for such profile comparisons (HPO phenotype sets, GO annotations).
</details>

**Exercise 8 (★★★).** Build the Wang similarity on the **Disease Ontology** instead of MONDO for AD1 versus late-onset Parkinson disease (map OMIM via `MIM:` xrefs), and report DO's coverage of Fdataset. Why is MONDO the better choice for this project?

<details><summary>Solution</summary>

```python
doid_terms = {first(s, "id"): s for s in doid_live}
doid_parents = {t: [v for v, _ in s.get("is_a", []) if v in doid_terms] for t, s in doid_terms.items()}
omim2doid_map = defaultdict(list)
for t, s in doid_terms.items():
    for v, _ in s.get("xref", []):
        if v.startswith("MIM:"):
            omim2doid_map[v[4:]].append(t)
a, b = omim2doid_map.get("104300", []), omim2doid_map.get("168600", [])
print("104300 ->", a, " 168600 ->", b)
if a and b:
    print("DO Wang:", round(wang_similarity([a, b], doid_parents)[0, 1], 4))
F_omims = [str(x[0])[1:] for x in sio.loadmat("data/raw/benchmarks/Fdataset.mat")["Wdname"].ravel()]
print(f"DO coverage of Fdataset: {np.mean([o in omim2doid_map for o in F_omims]):.1%}")
```

```text
104300 -> ['DOID:0080348']  168600 -> ['DOID:0060892']
DO Wang: 0.1027
DO coverage of Fdataset: 59.1%
```

DO places only about 59% of Fdataset's diseases (versus 97% for MONDO), because many benchmark diseases are specific OMIM genetic subtypes that DO does not model separately; 40% of diseases would have no `sem` neighbours at all. MONDO also has explicit, qualified equivalence mappings to OMIM. For the pairs DO does cover, the values differ from MONDO's because the hierarchies differ — a reminder that semantic similarity is relative to an ontology and a release.
</details>

**Exercise 9 (★★★).** Sensitivity to $w$: compute the Fdataset `sem_mondo` matrix for $w = 0.3$ and $w = 0.8$ and compare with $w = 0.5$ (Spearman correlation of all pairs, and the overlap of each disease's 10 nearest neighbours). Does the choice of $w$ matter for a $k$-NN graph?

<details><summary>Solution</summary>

```python
sets = [m.split("|") if m else [] for m in ds.mondo]
base = S_sem
def knn_sets(S, k=10):
    out = []
    for i in range(len(S)):
        row = np.nan_to_num(S[i].copy(), nan=-1.0); row[i] = -1
        out.append(set(np.argsort(-row)[:k]))
    return out
nb = knn_sets(base)
mapped = [i for i, s in enumerate(sets) if s]
for w in [0.3, 0.8]:
    Sw = wang_similarity(sets, parents, w=w)
    rho = spearmanr(Sw[iu][ok], base[iu][ok])[0]
    nw = knn_sets(Sw)
    overlap = np.mean([len(nw[i] & nb[i]) / 10 for i in mapped])
    print(f"w={w}: Spearman vs w=0.5 {rho:.3f}, mean 10-NN overlap {overlap:.3f}")
```

```text
w=0.3: Spearman vs w=0.5 0.995, mean 10-NN overlap 0.862
w=0.8: Spearman vs w=0.5 0.973, mean 10-NN overlap 0.732
```

The global rankings are almost unchanged (Spearman 0.995 for $w = 0.3$, 0.973 for $w = 0.8$), but the *top-10 lists* — which are what the graph uses — change noticeably: 86% of neighbours are kept at $w = 0.3$ and only 73% at $w = 0.8$. A larger $w$ gives more weight to distant shared ancestors and favours terms with rich multi-parent ancestry. So $w$ is a real hyperparameter of the view. The project keeps the literature default 0.5 instead of tuning it, because tuning on the evaluation folds would leak test information; if you want to tune it, do so on the separate validation split described in `HOW_IT_WORKS.md` Section 10.
</details>

**Exercise 10 (★).** The code stores `omim2mondo` as a list and takes a maximum over term pairs, yet in the current release no OMIM number maps to more than one live MONDO class. Is the generality wasted? Give three situations in which several terms per disease would occur.

<details><summary>Solution</summary>

Not wasted — it makes the code correct for inputs it does not control. Several terms per disease occur when: (1) **another release or ontology** is used — older MONDO releases, or DO/MeSH, where an OMIM entry can be linked to a specific term and a broader one (our own table has OMIM 104300 → `MESH:C536594|MESH:D000544`); (2) **non-equivalence xrefs** are included — a `relatedTo` or broad mapping alongside the equivalence (the project's parser does not read qualifiers, so any future `relatedTo` OMIM xref would be added); (3) **obsoletion and merging in progress** — an OMIM entry split into two MONDO classes or two entries merged; or (4) entities that are inherently multi-term, such as a drug's indications or a disease's phenotype profile. The max rule then picks the best-matching interpretation.
</details>

**Exercise 11 (★★).** MeSH: (a) Give the ancestors of tree number `F03.615.400.100`. (b) Why does Alzheimer Disease have three tree numbers, and what does that make MeSH as a graph? (c) Why should you store `D000544` rather than `C10.228.140.380.100`?

<details><summary>Solution</summary>

(a) Truncate one segment at a time: `F03.615.400` (Dementia) → `F03.615` (Neurocognitive Disorders) → `F03` (Mental Disorders) — verified in Section 11.9.
(b) It is classified as a brain disease (C10.228…), as a neurodegenerative tauopathy (C10.574…) and as a mental disorder (F03…). Each tree number lives in a tree, but the descriptor-level graph has multiple parents: a DAG.
(c) Tree numbers encode *positions*, which change when NLM reorganises the trees; descriptor IDs are stable identifiers. Derive tree numbers from the current MeSH release when you need them.
</details>

**Exercise 12 (★★★).** Shallow annotation: for the Fdataset diseases that have a MONDO term, compute the number of ancestors of each, list the five shallowest, and compare their mean `sem_mondo` similarity to all other diseases with that of the rest.

<details><summary>Solution</summary>

```python
n_anc = {i: len(ancestors(ds.mondo.iloc[i].split("|")[0])) for i in mapped}
shallow = sorted(mapped, key=n_anc.get)[:5]
for i in shallow:
    print(f"{n_anc[i]:3d} ancestors  OMIM {omims[i]}  {ds.name.iloc[i][:40]:40s} -> {name(ds.mondo.iloc[i])}")
mean_row = lambda i: np.nanmean(np.delete(S_sem[i], i))
print("mean similarity, 5 shallowest:", round(np.mean([mean_row(i) for i in shallow]), 3),
      "| all mapped:", round(np.mean([mean_row(i) for i in mapped]), 3))
print("median #ancestors over mapped diseases:", int(np.median(list(n_anc.values()))))

SUSC, DIS = "MONDO:0042489", "MONDO:0000001"          # the two roots involved
anc_of = {i: ancestors(ds.mondo.iloc[i]) for i in mapped}
only_susc = [i for i in mapped if SUSC in anc_of[i] and DIS not in anc_of[i]]
only_dis = [i for i in mapped if DIS in anc_of[i] and SUSC not in anc_of[i]]
both = [i for i in mapped if DIS in anc_of[i] and SUSC in anc_of[i]]
print(len(only_susc), "only under 'disease susceptibility' |", len(only_dis), "only under 'disease' |",
      len(both), "under both")
print("max similarity, susceptibility-only vs disease-only:", np.nanmax(S_sem[np.ix_(only_susc, only_dis)]))
j, k = omims.index("131200"), omims.index("140600")
print("endometriosis susceptibility vs osteoarthritis susceptibility:", round(S_sem[j, k], 4))
```

```text
  3 ancestors  OMIM 131200  Endometriosis, susceptibility to, 1      -> endometriosis, susceptibility to, 1
  3 ancestors  OMIM 140600  Osteoarthritis susceptibility 2          -> osteoarthritis susceptibility 2
  3 ancestors  OMIM 146500  Multiple system atrophy 1, susceptibilit -> multiple system atrophy 1, susceptibility to
  3 ancestors  OMIM 165720  Osteoarthritis susceptibility 1          -> osteoarthritis susceptibility 1
  3 ancestors  OMIM 166760  Otitis media, susceptibility to          -> otitis media, susceptibility to
mean similarity, 5 shallowest: 0.048 | all mapped: 0.127
median #ancestors over mapped diseases: 13
33 only under 'disease susceptibility' | 260 only under 'disease' | 11 under both
max similarity, susceptibility-only vs disease-only: 0.0
endometriosis susceptibility vs osteoarthritis susceptibility: 0.4286
```

The result is not what the "shallow annotation" story predicts, and it reveals a real problem. All five shallowest diseases are OMIM *susceptibility* entries. MONDO files them under its separate root *disease susceptibility*, as direct children of the single grouping class *inherited disease susceptibility* (three ancestors: itself, that class, the root). Consequences for `sem_mondo`:

* similarity **exactly 0** to the 260 diseases that sit only under the main *disease* root (the maximum printed above is 0.0) — so "osteoarthritis susceptibility 1" is not linked to any ordinary disease class at all; its non-zero similarities (mean row value 0.048 for the five) come only from other susceptibility entries and from the 11 diseases that MONDO classifies under both roots;
* similarity **0.4286** to every other susceptibility entry placed the same way, whatever the organ system, by the sibling formula with $L = 2$: $0.5(1 - 0.5^2)/(1 - 0.5^3) = 0.375/0.875$.

So 33 Fdataset diseases (10.5%) form an artificial cluster in the ontology view. In a cold-start setting, a held-out "susceptibility" disease would borrow indications from unrelated susceptibility entries. Possible fixes: map each susceptibility class to the disease it predisposes to (by name or cross-reference) and use that class; or set these rows to NaN so the model falls back on `pheno_mim`. The general lesson: always inspect the depth/ancestry distribution and the roots of your mapped terms before trusting an ontology view.
</details>

**Exercise 13 (★).** (a) Under the true path rule, which annotations are implied by annotating a patient's disease as *Alzheimer disease type 1*? (b) Suppose someone wrongly added `AD1 is_a psychiatric disorder` directly. What would happen to AD1's similarity to schizophrenia, and why can't the data tell you the edge is wrong?

<details><summary>Solution</summary>

(a) Every ancestor in $T_{AD1}$ (35 terms including itself): early-onset autosomal dominant AD, familial AD, Alzheimer disease, dementia, cognitive disorder, psychiatric disorder, tauopathy, neurodegenerative disease, amyloidosis, cerebrovascular disorder, hereditary disease, …, disease.
(b) *Psychiatric disorder* is already an ancestor of AD1, at distance 6 ($S = 0.0156$). A direct edge would make its distance 1 ($S = 0.5$) and its own ancestors closer too, so the shared part with schizophrenia (whose $S_{SCZ}$(psychiatric disorder) $= 0.125$) would jump, and the similarity would rise substantially. Nothing in the similarity computation can flag the edge: Wang's method trusts the graph completely. (The direct edge would also be redundant, since the relationship is already implied — ontology QC tools flag such redundant edges.)
</details>

---

## 15. Answers to the PREREQUISITES.md self-check questions (D3)

### Q1. In Wang's method, why does a close common ancestor contribute more than the root?

**Mechanically**, because of how S-values are defined. Each step upward multiplies the contribution by $w < 1$, keeping the best path, so an ancestor at shortest distance $d$ contributes $S_A(t) = w^{d}$ (Section 8.4). With $w = 0.5$ a parent contributes 0.5, a grandparent 0.25, and the root of MONDO — 7 or more steps above a typical OMIM disease — contributes $0.5^7 = 0.0078$ or less. A shared ancestor $t$ adds $S_A(t) + S_B(t)$ to the numerator, so a shared *parent* adds up to 1, while the shared root adds about 0.016. In the AD1/AD2 example, *familial Alzheimer disease* contributes 0.75 of the numerator 3.86 (19%), the root 0.016 (0.4%).

**Conceptually**, because the root carries no information. Every disease is a "disease"; sharing the root cannot distinguish a similar pair from a random pair. A close common ancestor means the two terms share almost their entire meaning — they differ only in the last refinement. In information-theoretic language, the root has information content ~0 and a specific ancestor high information content; Wang's distance decay is a topology-based stand-in for that IC gradient, with no annotation corpus needed.

**Quantitatively**, the closed form on a chain (Section 8.5), $\mathrm{Sim}_W \approx w^k$ where $k$ is the number of steps from each term to their lowest common ancestor, shows that similarity is governed almost entirely by *how close the LCA is*, not by how many distant ancestors are shared: siblings ≈ 0.5, cousins ≈ 0.25 for $w = 0.5$. This is exactly the behaviour we want from a disease-similarity view for guilt-by-association: drugs for "familial Alzheimer disease" are good candidates for another familial Alzheimer form; drugs for "nervous system disorders" in general are not.

**Caveat.** Because every term includes the root, Wang similarity has a floor above 0 within one root (AD1 vs type 2 diabetes 0.054; AD1 vs the root itself 0.165). The design makes the root's contribution small, not zero — another reason to use the values as ranks.

### Q2. Why can one OMIM disease map to several MONDO terms, and how do we handle that?

**Why it can happen.**

1. *Different data models.* OMIM entries are organised around genes and loci; MONDO classes are organised around disease concepts. An OMIM phenotype entry can correspond to a MONDO class *and* be related to a broader grouping class (e.g. the phenotypic-series class), or one OMIM entry may cover what MONDO models as two clinical entities.
2. *Non-equivalence cross-references.* An `xref` line can mean "related", not "same". MONDO qualifies its xrefs (`equivalentTo`, `relatedTo`, `obsoleteEquivalent`, …). A parser that, like ours, takes every `OMIM:` xref regardless of qualifier would collect a `relatedTo` mapping as a second term.
3. *Curation in progress.* Merges, splits and obsoletions happen every release; transiently, two live classes can carry the same OMIM xref.
4. *Other ontologies.* In MeSH, DO or older resources one-to-many links are common (our own table maps OMIM 104300 to two MeSH IDs).

**What the data say now.** MONDO's equivalence axioms are designed to be one-to-one — two MONDO classes both *equivalent* to one OMIM entry would be logically equivalent to each other. In the 2026-09-01 release, none of the 9,904 OMIM numbers referenced by live MONDO classes maps to more than one class, and all OMIM xrefs carry equivalence-type qualifiers (Section 11.8). So for this release every mapped benchmark disease has exactly one MONDO term. 11 of 415 have none (obsolete or never-equivalent OMIM entries) and get a `NaN` row.

**How the project handles it.**

* `parse_mondo` stores `omim2mondo` as a **list** per OMIM number, and `disease_table` writes all of them, `|`-separated, into `diseases_all.csv`.
* `wang_similarity` takes a list of terms per disease and uses the **maximum** pairwise term similarity: $\mathrm{sim}(X, Y) = \max_{a \in X, b \in Y} \mathrm{Sim}_W(a, b)$.
* Rationale: multiple MONDO terms for one OMIM entry are *alternative codings of the same disease*, so the best-matching coding is the meaningful one. Averaging would penalise a disease for having been mapped twice (it would not even be fully similar to itself); BMA is designed for entities with several genuinely different aspects (phenotype profiles), which is not the situation here (Section 9).
* Entities with **no** term get `NaN`, i.e. no `sem_mondo` neighbours; the model relies on `pheno_mim` (which covers every disease) and `gene_d` for them.

A more careful variant would read the qualifiers and keep only `equivalentTo` mappings, falling back to `relatedTo` only when no equivalence exists.

---

## 16. Summary and cheat sheet

**Ontology vocabulary.** Class/term; ID as CURIE (`MONDO:0007088`) ↔ IRI (`http://purl.obolibrary.org/obo/MONDO_0007088`); label (`name`); synonyms with scope EXACT/BROAD/NARROW/RELATED; `def`; `xref` (with qualifiers in MONDO); `subset`; `is_obsolete` + `replaced_by`/`consider`.

**Relations.** `is_a` (subsumption, transitive, inheritance); `part_of` (parthood, transitive); others via `relationship:`. True path rule: annotation to a term implies annotation to all ancestors.

**Graph facts.** DAG, not tree: multiple parents → multiple paths, depths, LCAs. Ancestors include self (convention). MICA = common ancestor with maximum IC.

**OBO format.** Header tags; `[Term]`/`[Typedef]` stanzas; `tag: value {modifiers} ! comment`. Parse line by line; filter prefix and obsolete; keep parents that exist. OBO ≈ a readable serialisation of OWL; OWL adds logical definitions and reasoning.

**Disease resources.**

| Resource | IDs | Structure | Note |
|---|---|---|---|
| MeSH | `D######`, SCR `C######` | thesaurus, tree numbers (multiple per descriptor) | literature indexing; tree numbers unstable |
| DO | `DOID:####` | ontology (DAG) | OMIM xrefs as `MIM:`; ~59% of F diseases |
| OMIM | six-digit MIM numbers (`*`, `#`, `%`, `+`, `^`) | flat catalogue | first digit = inheritance/date class |
| MONDO | `MONDO:#######` | merged ontology, qualified equivalences | ~97% of F diseases; 1:1 OMIM equivalence |
| HPO | `HP:#######` | phenotype ontology + disease annotations | phenotype-profile similarity |

**Similarity measures.**

| Family | Measure | Formula |
|---|---|---|
| Edge | path | shortest path through a common ancestor |
| Edge | Wu–Palmer | $2\,\mathrm{depth(LCA)}/(\mathrm{depth}(a) + \mathrm{depth}(b))$ |
| IC | Resnik | $\mathrm{IC}(\mathrm{MICA})$ |
| IC | Lin | $2\,\mathrm{IC}(\mathrm{MICA})/(\mathrm{IC}(a) + \mathrm{IC}(b))$ |
| IC | Jiang–Conrath | distance $\mathrm{IC}(a) + \mathrm{IC}(b) - 2\,\mathrm{IC}(\mathrm{MICA})$ |
| Hybrid | Wang | $\sum_{t \in T_A \cap T_B}(S_A(t) + S_B(t)) / (SV(A) + SV(B))$, $S_A(t) = w^{\text{shortest distance}}$ |

IC: $-\log p(t)$; intrinsic $p(t) = |\mathcal{D}(t)|/N$ or corpus-based. Wang with $w = 0.5$: parent–child → 0.75, siblings → 0.5, cousins → 0.25. Term sets: max (alternative codings), BMA (profiles), average (rarely right).

**Project facts.** `sem_mondo` = Wang, $w = 0.5$, MONDO `is_a` only, max over mapped terms, NaN if unmapped. MONDO 2026-09-01: 32,109 live classes, 46,582 `is_a` edges, 10,722 multi-parent classes, 4 roots. 33 Fdataset diseases sit only under the *disease susceptibility* root (zero similarity to ordinary diseases, ~0.43 to each other) — a known weakness of the view. Coverage 97.1% (F) / 97.3% (C) vs DO 59.1% / 64.5%. Cold-start AUPR (F): `pheno_mim` 0.149, `sem_mondo` 0.106, both 0.175.

---

## 17. Further resources (all links checked in October 2026)

DOI links resolve to the publisher's page; some publishers (ACS, OUP, medRxiv, OMIM) refuse automated link checkers but open normally in a browser. "Free" means readable without a subscription.

**Ontology foundations and formats (free)**

* OBO Foundry (free): https://obofoundry.org/ — the community of interoperable biomedical ontologies; principles, registry, links to every ontology here.
* OBO flat file format 1.4 syntax and semantics (free): https://owlcollab.github.io/oboformat/doc/obo-syntax.html — the formal specification, including the mapping to OWL.
* OBO 1.2 format guide (free): https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html — the older, very readable guide to tags and stanzas.
* OWL 2 Web Ontology Language Primer (free): https://www.w3.org/TR/owl2-primer/ — the gentlest official introduction to OWL.
* Gene Ontology — relations and the true path rule (free): https://geneontology.org/docs/ontology-relations/ — `is_a`, `part_of` and how relations compose.
* SSSOM — Simple Standard for Sharing Ontological Mappings (free): https://mapping-commons.github.io/sssom/ — how to express and exchange mappings with explicit predicates.
* Bioregistry (free): https://bioregistry.io/ — canonical prefixes and their synonyms (`MIM`/`OMIM`, `MSH`/`MESH`).

**Disease resources (free)**

* MONDO website (free): https://mondo.monarchinitiative.org/ — downloads, documentation, design principles.
* MONDO documentation (free): https://mondo.readthedocs.io/en/latest/ — mapping qualifiers, obsoletion policy, how to request changes.
* MONDO on GitHub (free): https://github.com/monarch-initiative/mondo — releases and issue tracker (where true-path violations get fixed).
* MONDO in the EBI Ontology Lookup Service (free): https://www.ebi.ac.uk/ols4/ontologies/mondo — browse terms and hierarchies interactively.
* Vasilevsky et al. (2022), "Mondo: Unifying diseases for the world, by the world", *medRxiv* (free): https://doi.org/10.1101/2022.04.13.22273750.
* Human Disease Ontology (free): https://disease-ontology.org/ ; Schriml et al. (2022), "The Human Disease Ontology 2022 update", *Nucleic Acids Res.* 50:D1255 (free): https://doi.org/10.1093/nar/gkab1063.
* MeSH home page (free): https://www.nlm.nih.gov/mesh/meshhome.html ; MeSH browser (free): https://meshb.nlm.nih.gov/ ; "MeSH tree structures" explainer (free): https://www.nlm.nih.gov/mesh/intro_trees.html.
* OMIM (free for non-commercial browsing; bulk downloads need registration): https://www.omim.org/ — the site blocks automated link checkers, but is the canonical source. Amberger et al. (2019), "OMIM.org: leveraging knowledge across phenotype–gene relationships", *Nucleic Acids Res.* 47:D1038 (free): https://doi.org/10.1093/nar/gky1151.
* Human Phenotype Ontology (free): https://hpo.jax.org/ ; Gargano et al. (2024), "The Human Phenotype Ontology in 2024", *Nucleic Acids Res.* 52:D1333 (free): https://doi.org/10.1093/nar/gkad1005.

**Semantic similarity — papers**

* Wang, Du, Payattakool, Yu & Chen (2007), "A new method to measure the semantic similarity of GO terms", *Bioinformatics* 23:1274 (publisher site; free access varies by institution): https://doi.org/10.1093/bioinformatics/btm087 — the method used for `sem_mondo`.
* Wang, Wang, Lu, Song & Cui (2010), "Inferring the human microRNA functional similarity and functional network based on microRNA-associated diseases", *Bioinformatics* 26:1644 (publisher site; access varies): https://doi.org/10.1093/bioinformatics/btq241 — Wang's method on the MeSH disease DAG with $w = 0.5$.
* Pesquita, Faria, Falcão, Lord & Couto (2009), "Semantic similarity in biomedical ontologies", *PLoS Comput. Biol.* 5:e1000443 (free): https://doi.org/10.1371/journal.pcbi.1000443 — the best survey; read it after this chapter.
* Resnik (1995), "Using information content to evaluate semantic similarity in a taxonomy", IJCAI (free preprint): https://arxiv.org/abs/cmp-lg/9511007 ; extended version, Resnik (1999), *JAIR* 11:95 (free): https://arxiv.org/abs/1105.5444.
* Lin (1998), "An information-theoretic definition of similarity", ICML (free PDF via Semantic Scholar): https://www.semanticscholar.org/paper/An-Information-Theoretic-Definition-of-Similarity-Lin/cc0c3033ea7d4e19e1f5ac71934759507e126162 ; bibliographic record: https://dblp.uni-trier.de/rec/html/conf/icml/Lin98.
* Jiang & Conrath (1997), "Semantic similarity based on corpus statistics and lexical taxonomy" (free preprint): https://arxiv.org/abs/cmp-lg/9709008.
* Köhler et al. (2009), "Clinical diagnostics in human genetics with semantic similarity searches in ontologies", *Am. J. Hum. Genet.* 85:457 (free): https://doi.org/10.1016/j.ajhg.2009.09.003 — HPO + Resnik, the Phenomizer.
* van Driel, Bruggeman, Vriend, Brunner & Leunissen (2006), "A text-mining analysis of the human phenome", *Eur. J. Hum. Genet.* 14:535 (paid): https://doi.org/10.1038/sj.ejhg.5201585 — MimMiner, the source of `pheno_mim`.
* Yu, Wang, Yan, He & He (2015), "DOSE: an R/Bioconductor package for disease ontology semantic and enrichment analysis", *Bioinformatics* 31:608 (publisher site; access varies): https://doi.org/10.1093/bioinformatics/btu684 — a reference implementation of Wang/Resnik/Lin on DO.

**Software (free)**

* GOSemSim vignette (free): https://bioconductor.org/packages/release/bioc/vignettes/GOSemSim/inst/doc/GOSemSim.html — clear worked descriptions of IC- and graph-based measures with code.
* pronto (Python OBO/OWL parser, free): https://pronto.readthedocs.io/
* obonet (OBO → NetworkX, free): https://github.com/dhimmel/obonet
* Ontology Access Kit, OAK (free): https://incatools.github.io/ontology-access-kit/ — one Python API for many ontologies, including semantic similarity.

---

## 18. Glossary

* **Ancestor** — any term reachable by following parent edges (here including the term itself).
* **Annotation** — a link from a real entity (gene, disease record, patient) to an ontology class.
* **BMA (best-match average)** — set similarity averaging each term's best match in the other set.
* **Class / term** — a named category in an ontology.
* **Corpus-based IC** — information content estimated from annotation frequencies.
* **CURIE** — compact URI, `PREFIX:local_id`.
* **Cross-reference (xref)** — link from a term to an identifier in another resource.
* **DAG** — directed acyclic graph; allows multiple parents, forbids cycles.
* **Depth (min/max)** — shortest/longest path length from a term to a root.
* **Descendant** — any term from which the given term is reachable upward.
* **Disease Ontology (DO, DOID)** — OBO Foundry ontology of human diseases.
* **Equivalence mapping** — cross-reference asserting two identifiers denote the same class (`MONDO:equivalentTo`, `skos:exactMatch`).
* **HPO** — Human Phenotype Ontology of phenotypic abnormalities.
* **Information content (IC)** — $-\log p(t)$; specificity of a term.
* **Intrinsic IC** — IC estimated from the ontology structure (descendant counts).
* **IRI** — internationalised resource identifier; the full web identifier of a term.
* **`is_a`** — subclass relation; transitive; supports inheritance.
* **Jiang–Conrath distance** — $\mathrm{IC}(a) + \mathrm{IC}(b) - 2\,\mathrm{IC}(\mathrm{MICA})$.
* **LCA** — lowest common ancestor(s); several possible in a DAG.
* **Lin similarity** — $2\,\mathrm{IC}(\mathrm{MICA})/(\mathrm{IC}(a) + \mathrm{IC}(b))$.
* **MeSH** — NLM's Medical Subject Headings thesaurus; descriptors, SCRs, tree numbers.
* **MICA** — most informative common ancestor.
* **MimMiner** — text-mined phenotype similarity of OMIM records (van Driel et al., 2006).
* **MIM number** — six-digit OMIM identifier.
* **MONDO** — Mondo Disease Ontology, merging OMIM, Orphanet, DO, NCIt, MeSH and others.
* **OBO format** — line-based ontology file format with stanzas and tag–value pairs.
* **Obsolete term** — retired class kept for traceability (`is_obsolete: true`).
* **OMIM** — Online Mendelian Inheritance in Man, catalogue of genes and genetic phenotypes.
* **Ontology** — formal specification of classes and typed relations in a domain.
* **OWL** — W3C Web Ontology Language, logic-based; OBO is a serialisation of a subset.
* **`part_of`** — parthood relation (BFO:0000050).
* **Phenotypic series (PS)** — OMIM grouping of genetically heterogeneous forms of one phenotype.
* **Resnik similarity** — IC of the MICA.
* **S-value** — Wang's semantic contribution $S_A(t)$ of ancestor $t$ to term $A$.
* **Semantic similarity** — similarity of meaning computed from an ontology.
* **Semantic value (SV)** — sum of a term's S-values.
* **SSSOM** — standard format for ontology mappings.
* **Stanza** — one `[Term]`/`[Typedef]` block in an OBO file.
* **Synonym scope** — EXACT, BROAD, NARROW or RELATED.
* **Tree number** — MeSH dotted code giving a descriptor's position in a tree.
* **True path rule** — annotation to a term implies annotation to all its ancestors.
* **Wang similarity** — weighted overlap of ancestor sets with S-values decaying by $w$ per edge.
* **Wu–Palmer similarity** — $2\,\mathrm{depth(LCA)}/(\mathrm{depth}(a) + \mathrm{depth}(b))$.
