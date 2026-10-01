# Unit D4 — Biomedical Databases and Entity Resolution

*Every node in our graph is a row in someone else's database. This chapter teaches you how those databases work, how their identifiers behave, and how to join them without quietly corrupting your data.*

---

## 0. About this chapter

**Prerequisites.**
- **A1 (Python / NumPy / pandas)**: reading CSV/TSV files, `groupby`, `merge`, dictionaries and sets.
- **D1 (Pharmacology and drug repositioning)**: what an indication is, what a drug target is.
- **D2 (Cheminformatics)**: SMILES, molecular graphs, fingerprints. You need the idea that a molecule can be written as a string.
- **D3 (Ontologies)**: ontologies as DAGs, the OBO format, `is_a`, MONDO, MeSH trees. This chapter re-uses those ideas for *identifier mapping* rather than for similarity.
- Helpful: a little HTTP (what a URL, a GET request and a status code are). Section 2.6 explains what you need.

**Estimated study time.** 7–9 hours: about 3.5 h on the theory (Sections 1–3), 2 h running the code (Section 4), 1 h on the project walk-through (Section 5) and 2–3 h on the exercises.

**How to run the code.** Every code block in Section 4 was executed from the project root with the project environment, and the output shown under it is the real output:

```powershell
cd "C:\Users\Abhineet Anand\Desktop\DrugRepositioning"
.venv\Scripts\python.exe        # then paste a block, or save it as a .py file and run it
```

The blocks read only the files already in `data/raw/` and `data/interim/`. Only Section 4.9 touches the internet, and it sends exactly two requests.

**Learning objectives.** After this unit you will be able to:

1. Name the main biomedical databases used in drug repositioning (DrugBank, PubChem, ChEMBL, UniChem, OMIM, MeSH, CTD, MedGen/UMLS, MONDO/DO, ClinicalTrials.gov, KEGG, UniProt, NCBI Gene, HGNC, CMap/LINCS), say what each contains, and recognise each one's identifiers on sight.
2. Explain the difference between a PubChem **Substance** (SID), **Compound** (CID) and **BioAssay** (AID), and why DrugBank IDs can be resolved through PubChem.
3. Write correct PUG-REST and ClinicalTrials.gov API v2 URLs, and call them politely (rate limits, retries, caching).
4. Decode an InChIKey into its three blocks and use the first block to detect stereoisomers, and explain when a salt and its parent get the same key.
5. Read an OMIM number (`#104300`, `*104760`, `%`, `+`, `^`) and say whether it is a gene or a phenotype entry and whether the entry is still live.
6. Distinguish MeSH descriptors (`D…`), supplementary concept records (`C…`) and tree numbers, and explain why tree numbers are not identifiers.
7. Explain CTD's **curated** versus **inferred** links, its `DirectEvidence` values (`marker/mechanism`, `therapeutic`), the MEDIC disease vocabulary, and how each CTD file is laid out.
8. Explain why CTD inferred and therapeutic gene–disease links would leak the answer into a drug-repositioning model.
9. Design an entity-resolution cascade (exact cross-references → structure keys → names), measure its coverage, and detect conflicts and many-to-many mappings in code.
10. Recognise the classic entity-resolution pitfalls: retired and merged IDs, namespace collisions, prefix formats, empty-key joins, granularity mismatch, version drift.
11. Record provenance (download date, internal version, checksum, licence) for every file a project uses, and state the licence terms of each source used here.
12. Read `scripts/01_download_data.py`, `scripts/02_build_features.py` and `scripts/06_case_study.py` and explain every mapping decision, including two silent problems that this chapter uncovers.

---

## 1. Motivation: one drug, one disease, seven databases

Our benchmark files (`Fdataset.mat`, `Cdataset.mat`) give us a 0/1 matrix plus two lists of identifiers: DrugBank IDs for the rows (e.g. `DB00563`) and OMIM numbers for the columns (e.g. `104300`). That is all. Everything else the model uses — chemical structures, gene profiles, a disease hierarchy, readable names for the case studies, independent evidence for checking predictions — lives in *other* databases that use *other* identifiers. Before any graph neural network can run, we must answer questions like:

> *Which PubChem compound is DrugBank's `DB00563`? Which CTD chemical is that compound? Which MONDO term, which MeSH term and which CTD disease is OMIM `104300`?*

Let us follow the drug **methotrexate** (`DB00563`) and the disease **Alzheimer disease type 1** (OMIM `104300`) through the project's pipeline. All the values below are real and come from the project's files.

```
DB00563 ──PubChem substance deposited by DrugBank──► CID 126941 ──► SMILES, InChIKey, name "Methotrexate"
        ──CTD chemical vocabulary, by name───────────► MESH:D008727 "Methotrexate"  ──► human genes (CTD)

OMIM 104300 ──MedGen (src = OMIM)──► CUI C1863052 "Alzheimer disease type 1"
            ──MONDO xref (equivalentTo)──► MONDO:0007088 ──► position in the disease hierarchy
            ──MONDO xref──► MESH:C536594 (a supplementary concept)  ┐
            ──CTD MEDIC alt-ID──► MESH:D000544 "Alzheimer Disease"  ┴► CTD disease genes (104 genes)
```

Seven databases for one row and one column. Now consider two things that can go wrong, both of which this chapter will show really happened in this project's data:

1. **A silent format mismatch.** CTD's chemical vocabulary stores PubChem identifiers as the string `CID:126941`, while PubChem returns the bare number `126941`. A join on these columns matches **nothing** — and raises no error. The project's code tries "CID first, then InChIKey, then name"; because of this format difference (and because the current CTD file has no InChIKeys at all) every one of its 643 drug-to-CTD mappings actually came from the *name* step (Section 4.7).
2. **A cross-reference that points to the wrong thing.** If you do fix the prefix, CTD's CID column maps methotrexate's CID 126941 not to the methotrexate record but to the record **"Folic Acid Antagonists"** — a drug *class*. Lidocaine's CID points to "Lidoderm" (a commercial patch product), rosiglitazone's to "rosiglitazone-metformin combination". An "exact ID match" is only as good as the cross-reference behind it.

Neither problem produces a crash, a warning or an obviously wrong number. They produce a *plausible but subtly different* dataset. That is the central lesson of this unit: **entity resolution fails silently, so you must measure it.** By the end you will be able to build mapping tables that report their own coverage, cardinality and conflicts, and to explain to a reviewer exactly where every identifier in the project came from and under what licence.

---

## 2. Core theory

### 2.1 Identifiers: the grammar of biomedical data

**Intuition.** A database is a collection of *records* about *entities* (a drug, a gene, a disease). Each record has an **identifier** (ID) — a short string whose only job is to point at that record, forever, without ambiguity. Names are for humans; IDs are for joins.

**Precise vocabulary.**

| Term | Meaning | Example |
|---|---|---|
| **Namespace** (prefix) | The authority that issued the ID | `MESH`, `OMIM`, `CHEBI`, `MONDO` |
| **Local ID** | The part that is unique *within* the namespace | `D008727`, `104300` |
| **CURIE** (compact URI) | `prefix:local_id`, unique across namespaces | `MESH:D008727`, `OMIM:104300` |
| **Primary ID** | The current, preferred ID of a record | `DB00943` |
| **Secondary / alternative ID** | An older or merged ID that still resolves to the record | `APRD00562` (legacy DrugBank ID of zalcitabine) |
| **Obsolete / retired ID** | An ID whose record was withdrawn, merged or split | OMIM entries marked `^` |
| **Cross-reference (xref)** | A statement in database X that its record corresponds to record Y in database Z | `xref: OMIM:104300` in a MONDO term |
| **Mapping qualifier** | How close the correspondence is | `equivalentTo`, `relatedTo`, `broadMatch` |

Three properties make an identifier good: it is **unique** (one ID, one entity), **stable** (it never changes meaning, and is never reused) and **resolvable** (you can look it up). Names fail all three: "AD" means Alzheimer disease, atopic dermatitis or autosomal dominant; "heparin" is a family of molecules; brand names change across countries.

**Bare numbers are dangerous.** `104300` is an OMIM number, but MedGen also issues its own internal numbers, and the same string can occur in both namespaces with unrelated meanings. Section 4.8 shows a real case where the bare string `602439` is OMIM's number for one of our diseases *and* MedGen's internal ID for "Voice testing inconclusive". Always carry the namespace with the number (`OMIM:602439`), or filter on the namespace column before joining.

**Not all xrefs are equal.** A cross-reference is a *claim* made by a curator (or by a program) at some date. Ontologies like MONDO attach a qualifier to each one: `equivalentTo` (same concept), `relatedTo` (overlapping, not the same), `equivalentObsolete` (equivalent, but the target ID has since been retired). Treat an xref without a qualifier as "probably related", not "identical".

### 2.2 Chemical databases

#### 2.2.1 DrugBank

**What it is.** DrugBank is a curated knowledge base about drugs: approved small molecules, biologics (proteins, peptides, antibodies, vaccines), nutraceuticals, experimental and withdrawn compounds. Each drug record (a "drug card") holds names and synonyms, chemical structure, the drug's *groups* (`approved`, `investigational`, `experimental`, `withdrawn`, `illicit`, `nutraceutical`, `vet_approved`), pharmacology (indication text, mechanism, pharmacodynamics), targets, enzymes, transporters and carriers (with UniProt accessions), drug–drug interactions, products and external links. It began in David Wishart's group at the University of Alberta and is now maintained by a company (DrugBank / OMx Personalized Health). The current major description is Knox et al. (2024), *DrugBank 6.0*.

**Identifiers.**
- **Primary drug ID:** `DB` followed by five digits, e.g. `DB00945` (aspirin), `DB00563` (methotrexate). This is what the benchmarks use.
- **Secondary IDs:** older accession schemes kept so that old links still resolve (e.g. `APRD00562` for zalcitabine, whose primary ID is `DB00943`).
- **Salts:** a drug record describes the *active moiety*; its salt forms (e.g. a hydrochloride, a sodium salt) are listed inside the parent record, each with its own `DBSALT…` identifier. This matters because two benchmark rows can never be "the same drug in a different salt" — DrugBank already merged salts into one card — but *other* databases (PubChem substances, CTD chemicals) may list salts separately.
- **Biologics** have sequences rather than small-molecule structures. Many have no PubChem *compound* at all (Section 2.2.2), which is why a few of our drugs (sermorelin, glatiramer, heparins, conjugated estrogens) lack SMILES.

**Versions.** DrugBank is released as numbered versions (5.1.x). At the time of writing the latest is **5.1.22, released 2026-06-27**. Records are added, merged and occasionally withdrawn between versions; three of our benchmark IDs (`DB00510`, `DB01258`, `DB01402`) returned nothing from PubChem; HOW_IT_WORKS treats them as retired DrugBank IDs, and they keep their ID as their name in our tables.

**Licensing — read this carefully.**
- The full database (XML, structures, target links…) is released under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** and, for academics, requires a free account and an approved application. Commercial use needs a paid licence.
- A small **"Open Data"** subset — the **DrugBank Vocabulary** (IDs, names, synonyms, CAS numbers, UNIIs, InChIKeys) and **DrugBank Structures** — is released under **CC0** (public domain). You may use it for anything.
- In 2026 DrugBank **paused academic downloads** while changing its distribution system. That is exactly why this project could not obtain the XML and had to resolve DrugBank IDs through PubChem.

> **Rule of thumb.** If you use DrugBank data in a paper, cite DrugBank and state the version. If you redistribute derived files (e.g. on GitHub), you may share IDs and anything derived from the CC0 vocabulary freely, but not bulk content from the CC BY-NC dataset for commercial purposes.

#### 2.2.2 PubChem: Substance vs Compound vs BioAssay

PubChem (run by the US National Center for Biotechnology Information, NCBI) is the largest free chemistry database. Its design is the key to understanding why our DrugBank lookup works.

**Three databases in one.**

| Database | ID | What one record is | Who creates it |
|---|---|---|---|
| **Substance** | **SID** | One *deposited* description of a chemical, exactly as a depositor sent it (may include a salt, a mixture, a drawing error) | The **depositor** (a "source": DrugBank, ChEMBL, a vendor, a journal…) |
| **Compound** | **CID** | One *unique standardized structure*, computed by PubChem from substances | PubChem, by **standardization** of substances |
| **BioAssay** | **AID** | One biological experiment and its results per substance | Depositors (screening centres, ChEMBL…) |

```
 depositors          Substance (SID)                    Compound (CID)
 ──────────          ───────────────                    ──────────────
 DrugBank   ──►  SID for "DB00945" (aspirin)   ─┐
 ChEMBL     ──►  SID for "CHEMBL25"            ─┼──standardize──►  CID 2244  (aspirin)
 a vendor   ──►  SID for "catalogue #A123"     ─┘
                                                     each SID points to ≥1 CID;
                                                     one CID collects many SIDs
```

Each depositor keeps its *own* identifier for each record, called the **source ID** (or "registry ID"). DrugBank deposits every drug card as a PubChem substance whose source ID is the DrugBank ID. That gives us a free, licence-friendly bridge:

$$\text{DrugBank ID} \xrightarrow{\text{source ID lookup}} \text{SID} \xrightarrow{\text{standardization}} \text{CID} \xrightarrow{\text{properties}} \text{SMILES, InChIKey, title}.$$

**Standardization and the parent compound.** When a substance is a salt or a mixture (e.g. metformin *hydrochloride*), PubChem creates a CID for the whole thing and also links it to its **components** and to its **parent** (the main organic fragment, neutralized). PUG-REST lets you choose: for SID input, `cids_type=standardized|component|all`; for CID input, `cids_type=parent` (and other relations such as `same_connectivity`, `same_stereo`). This is the formal version of HOW_IT_WORKS assumption 4: *"the PubChem compound is the parent compound (salts are stripped)"* — strictly, that is true when DrugBank deposited the active moiety, which is its convention.

**Scale.** The PubChem 2025 update (Kim et al., 2025) reports 119 million compounds, 322 million substances and 295 million bioactivity data points from more than 1,000 sources (counts as of September 2024). You will never download "all of PubChem" for a project like ours; you query it.

#### 2.2.3 PUG-REST: talking to PubChem by URL

**PUG-REST** ("Power User Gateway, REST style") encodes a whole query in the URL path, in three parts after a fixed prolog:

```
https://pubchem.ncbi.nlm.nih.gov/rest/pug  /  <input>                        /  <operation>            /  <output>
                                            compound/cid/2244                 property/InChIKey          TXT
                                            substance/sourceid/DrugBank/DB00945  cids                    TXT
```

- **Input** = domain (`compound`, `substance`, `assay`, `gene`, `protein`, …) + namespace (`cid`, `name`, `smiles`, `inchikey`, `sid`, `sourceid/<source name>`, `xref/…`) + identifiers (comma-separated for many).
- **Operation** = what to retrieve: `record`, `property/<list>`, `synonyms`, `cids`, `sids`, `aids`, `xrefs/<list>`, `description`, `classification`.
- **Output** = `JSON`, `XML`, `CSV`, `TXT`, `SDF`, `PNG`.

**Worked URL examples** (all are real, valid PUG-REST requests):

| Goal | URL (after `https://pubchem.ncbi.nlm.nih.gov/rest/pug`) | Returns |
|---|---|---|
| CID(s) for a DrugBank ID (the project's main call) | `/substance/sourceid/DrugBank/DB00945/cids/TXT` | `2244` |
| Synonyms of DrugBank's own substance (first line = DrugBank's name) | `/substance/sourceid/DrugBank/DB00945/synonyms/TXT` | `Aspirin`, … |
| Structure + key + title for up to ~100 CIDs at once | `/compound/cid/2244,126941/property/SMILES,InChIKey,Title/CSV` | a CSV table |
| CID from an InChIKey | `/compound/inchikey/BSYNRYMUTXBXSQ-UHFFFAOYSA-N/cids/TXT` | `2244` |
| CID(s) from a name (may be several!) | `/compound/name/methotrexate/cids/TXT` | one or more CIDs |
| Parent compound of a salt CID | `/compound/cid/<CID>/cids/TXT?cids_type=parent` | the parent CID |
| Which sources deposited records for a CID | `/compound/cid/2244/xrefs/SourceName/JSON` | list of depositors |

**Rules of the road.**
- **Rate limit:** PubChem asks every user, application or organization to send **no more than 5 requests per second**; heavier use is throttled. Each response carries an `X-Throttling-Control` header reporting your request-count and request-time status as Green/Yellow/Red/Black. Exceeding limits returns **HTTP 503**; persistent abuse gets your IP blocked for a growing period.
- **Timeout:** a synchronous PUG-REST request may run at most **30 seconds**.
- **Status codes:** `200` OK, `404` not found (e.g. a retired DrugBank ID), `400` malformed URL, `503` busy/throttled — retry later with a back-off.
- **Batch, cache, and be idempotent.** Ask for 100 CIDs in one property request instead of 100 requests; save results to disk (the project caches to `data/interim/pubchem_drugs.csv`) so that a re-run makes zero requests.
- Special characters in names or SMILES must be URL-encoded; long inputs can go in a POST body.

#### 2.2.4 InChI and the InChIKey

**InChI** (IUPAC International Chemical Identifier) is a *canonical* text representation of a structure, built in layers: formula, connectivity, hydrogens, charge/protonation, stereochemistry (double-bond and tetrahedral), isotopes. "Canonical" means that every correct drawing of the same molecule yields the same InChI, whatever atom order or drawing style was used. **Standard InChI** fixes all the options (tautomer handling, stereo perception) so that everyone's software produces the same string.

The **InChIKey** is a fixed-length, 27-character hash of the standard InChI, designed for database lookup and web search:

```
 BSYNRYMUTXBXSQ - UHFFFAOY S A - N
 └─── 14 ─────┘   └── 8 ─┘ │ │   │
   block 1:                │ │   └ block 3: protonation (N = neutral; M, O, … = proton removed/added)
   hash of the skeleton    │ └ InChI version (A = version 1)
   (connectivity layer)    └ S = standard InChI (N = non-standard)
                 block 2: hash of the remaining layers (stereo, isotopes, fixed H…)
```

Consequences that you will use constantly:

1. **Same molecule ⇒ same key**, regardless of source, drawing or file format. That is why InChIKey is the lingua franca for joining chemical databases (UniChem is built on it).
2. **Stereoisomers share block 1 and differ in block 2.** Omeprazole (racemic) and esomeprazole (its S-enantiomer) share `SUBDBMMJDZJVOS` but differ after the first hyphen. Section 4.6 finds 12 such skeleton-sharing groups among our 667 drugs.
3. **Salts are different molecules** to InChI: metformin and metformin hydrochloride get different keys. Strip the counter-ion first (keep the "parent" fragment) if you want them to match.
4. `UHFFFAOYSA` is the block-2 value you get when there is *no* stereo or isotope layer at all — a molecule with no defined stereocentres.
5. **Hashes can collide**, but for block 1 (a 14-character hash) the probability is negligible at database scale; the main practical risks are upstream: wrong structure deposited, undefined stereo, or tautomers that standard InChI treats differently.

A related idea: **CAS Registry Numbers** (e.g. `50-78-2` for aspirin) are widely used but proprietary (owned by the American Chemical Society); **UNII** codes (FDA) and **DTXSID** (US EPA CompTox) are free alternatives that appear in CTD's chemical file.

#### 2.2.5 ChEMBL and UniChem

**ChEMBL** (EMBL-EBI) is a manually curated database of *bioactivity*: measured potencies (IC50, Ki, EC50…) of molecules against targets, extracted from the medicinal-chemistry literature, patents and deposited screens, plus information on approved drugs, their mechanisms and indications. IDs look like `CHEMBL25` (aspirin, a *molecule*), and targets and assays have their own `CHEMBL…` IDs in the same namespace — always check *which* entity type a `CHEMBL` ID refers to. ChEMBL is free (CC BY-SA 3.0). For drug repositioning it is the best open source of drug–target potency and of ChEMBL's curated *drug indications* table — which, note, is itself a source of labels and must be treated as such (Unit E1).

**UniChem** (also EMBL-EBI) is a pure *identifier-mapping* service for chemicals. It links compound IDs across dozens of sources (ChEMBL, DrugBank, PubChem, ChEBI, KEGG, ZINC, …) using standard InChI/InChIKey as the hub. Querying UniChem with aspirin's InChIKey returns `CHEMBL25` from ChEMBL and `DB00945` from DrugBank in one call. It also offers "connectivity" searches that match on the skeleton (block 1), to collect salts, isotopes and stereoisomers of the same parent.

#### 2.2.6 KEGG (briefly)

KEGG (Kyoto Encyclopedia of Genes and Genomes) integrates pathways (`hsa04110`), genes (`hsa:5742`), compounds (`C00001`) and drugs (`D00109` = aspirin in KEGG DRUG). KEGG pathways are widely used for gene-set analyses. Licensing: the website is free for academic use, but bulk FTP access requires a paid subscription; KEGG's REST API (`rest.kegg.jp`) is for academic use with light traffic.

### 2.3 Genes and proteins (briefly)

Drug targets and disease genes live in their own identifier systems:

| System | ID example (human PTGS1, the COX-1 enzyme aspirin inhibits) | Notes |
|---|---|---|
| **NCBI Gene** ("Entrez Gene") | `5742` | Stable integer per gene per species; CTD's `GeneID` column |
| **HGNC** symbol / ID | `PTGS1` / `HGNC:9604` | The official human gene *symbol*; symbols change when genes are renamed — the ID does not |
| **UniProt** accession | `P23219` | A *protein* record; one gene can have several accessions; DrugBank targets use UniProt |
| **Ensembl** gene ID | `ENSG…` | Genome-annotation based; versioned (`.1`, `.2`) |

Two practical warnings. **Symbols are not IDs**: HGNC renames genes (e.g. to avoid symbols that Excel turns into dates), so a symbol from a 2012 file may not match a 2026 file; join on NCBI Gene IDs when you can. **Species matter**: CTD's chemical–gene file contains interactions from many organisms; the project keeps only `OrganismID == 9606` (human).

### 2.4 Disease vocabularies

#### 2.4.1 OMIM

**What it is.** OMIM (Online Mendelian Inheritance in Man), curated at Johns Hopkins University, is a catalogue of human genes and genetic phenotypes, focused on the relationship between genetic variation and phenotype. It is the continuation of Victor McKusick's book *Mendelian Inheritance in Man*.

**MIM numbers.** Every entry has a six-digit number whose first digit encodes history and inheritance:

| First digit | Meaning |
|---|---|
| 1, 2 | Autosomal loci or phenotypes, entries **created before 15 May 1994** |
| 3 | X-linked |
| 4 | Y-linked |
| 5 | Mitochondrial |
| 6 | Autosomal, entries **created after 15 May 1994** |

Allelic variants are written `entry.NNNN` (e.g. `300746.0001`).

**Entry symbols** (shown before the number on omim.org, *not* part of the number):

| Symbol | Meaning | Gene or phenotype? |
|---|---|---|
| `*` | A gene | gene |
| `#` | A descriptive phenotype entry whose molecular basis is known; the gene(s) are described in other entries | phenotype |
| `%` | A confirmed Mendelian phenotype or phenotypic locus whose molecular basis is **not** known | phenotype |
| `+` | A gene of known sequence **and** a phenotype in one entry (an older style) | both |
| (none) | A phenotype whose Mendelian basis is suspected but not established, or whose separateness is unclear | phenotype |
| `^` | The entry **no longer exists**: removed or moved to another entry | retired |

For drug repositioning, the benchmark columns are (almost all) **phenotype** entries (`#`, `%`, or no symbol): e.g. `#104300` *Alzheimer disease*. A gene entry such as `*104760` (APP, the amyloid precursor protein gene) is a different kind of object, and a model that treats a gene MIM number as a "disease" is wrong. Some of our 2011-era entries have since become `^` (Section 4.5).

**Phenotypic series** group genetically heterogeneous forms of one disorder (e.g. the many numbered Alzheimer types) under IDs that look like MIM numbers but are a separate namespace: `OMIMPS:…` in MONDO's xrefs. Do not mix `OMIMPS:104300`-style IDs with `OMIM:104300`.

**Licensing.** Johns Hopkins holds the copyright. The website is free to search. Of the bulk files, only `mim2gene.txt` (MIM number ↔ NCBI Gene / Ensembl / HGNC) is available without registration. Academic, non-profit and government users can register (free) for `mimTitles.txt`, `genemap2.txt` and `morbidmap.txt`; companies and anyone redistributing OMIM content in software need a licence. That is why the project takes OMIM *names* from NCBI MedGen instead (Section 2.4.3).

#### 2.4.2 MeSH

**What it is.** MeSH (Medical Subject Headings), from the US National Library of Medicine (NLM), is the controlled vocabulary used to index PubMed. It is updated **annually** (plus nightly additions of supplementary records).

**Record types.**
- **Descriptors** (main headings), IDs `D` + 6 digits (newer ones 9 digits): e.g. `D000544` *Alzheimer Disease*, `D008727` *Methotrexate*. Descriptors are arranged in 16 top-level categories (A anatomy, B organisms, **C diseases**, **D chemicals and drugs**, F psychiatry/psychology, …).
- **Qualifiers** (subheadings), e.g. "drug therapy", "genetics" — combined with descriptors in indexing.
- **Supplementary Concept Records (SCRs)**, IDs `C` + 6 digits (newer ones 9 digits): e.g. `C536594` *Alzheimer disease type 1*. SCRs are *not* in the tree; each is "heading-mapped" to one or more descriptors. Classes: 1 chemicals, 2 chemotherapy protocols, **3 diseases** (mostly rare diseases), 4 organisms (since 2018), 5 population groups (since 2023).

> **Do not confuse** MeSH's `C`-prefixed SCR IDs (`C536594`) with MeSH's **tree category C** (diseases) or with UMLS CUIs (`C1863052`). Three different meanings of a leading "C".

**Tree numbers.** A descriptor's position(s) in the hierarchy are written as dotted paths, e.g. *Alzheimer Disease* has three:
`C10.228.140.380.100` (nervous system diseases → … → dementia → Alzheimer), `C10.574.945.249` and `F03.615.400.100`. A concept can sit in several places (polyhierarchy). NLM states that tree numbers "have no intrinsic significance" and **change** when the hierarchy is revised. So: **join on descriptor/SCR IDs, never on tree numbers**; use tree numbers only to compute depth or ancestry *within one MeSH version*.

#### 2.4.3 UMLS and MedGen: concept unique identifiers

**UMLS** (Unified Medical Language System, NLM) merges more than a hundred vocabularies (MeSH, SNOMED CT, ICD, OMIM, HPO, …) into one *Metathesaurus*. Synonymous terms from different vocabularies are grouped into a **concept**, identified by a **CUI** (Concept Unique Identifier): `C` + 7 digits, e.g. `C0002395` *Alzheimer disease* (the general concept), `C1863052` *Alzheimer disease type 1*. UMLS itself requires a free licence (an account with the UMLS Terminology Services) because some source vocabularies carry restrictions.

**MedGen** (NCBI) is a free portal for conditions related to medical genetics. It integrates UMLS, OMIM, HPO, MONDO, Orphanet, GeneReviews and others into concepts, each with a CUI — the UMLS CUI where one exists, otherwise an NCBI-made one starting **`CN`** (e.g. `CN123456`). MedGen also uses an internal numeric UID (MONDO's xref `MEDGEN:354892` is that UID for C1863052).

The file the project uses, `MedGenIDMappings.txt.gz`, is a pipe-separated table:

```
#CUI_or_CN_id|pref_name|source_id|source|
C1863052|Alzheimer disease type 1|104300|OMIM|
C0002395|Alzheimer disease|104300|OMIM included|
```

The fourth column is crucial. `OMIM` means "this CUI *is* that OMIM entry"; `OMIM included` means "this concept is one of the *included titles* mentioned inside that OMIM entry" — a broader or narrower relative, not an equivalent. Section 4.4 shows that OMIM `104300` appears in four rows, only one of which is the equivalent concept.

#### 2.4.4 MONDO and the Disease Ontology

From Unit D3 you know MONDO (the Mondo Disease Ontology, Monarch Initiative) as a disease *hierarchy*. Here we care about its other job: it is the best open **cross-reference hub** for diseases. Each MONDO term lists xrefs to OMIM, Orphanet, DOID, MeSH, UMLS, MedGen, GARD, NCIT, ICD…, and each xref carries a provenance/qualifier annotation:

```
[Term]
id: MONDO:0007088
name: Alzheimer disease type 1
xref: DOID:0080348 {source="MONDO:equivalentTo"}
xref: MEDGEN:354892 {source="MONDO:equivalentTo", source="MONDO:MEDGEN"}
xref: MESH:C536594 {source="MONDO:equivalentTo"}
xref: OMIM:104300 {source="MONDO:equivalentTo", source="DOID:0080348"}
xref: UMLS:C1863052 {source="MEDGEN:354892", source="MONDO:equivalentTo", source="MONDO:MEDGEN"}
xref: DECIPHER:48 {source="...", source="MONDO:relatedTo"}
is_a: MONDO:0015140 ! early-onset autosomal dominant Alzheimer disease
```

(Excerpt from the project's `data/raw/ontology/mondo.obo`, `data-version: releases/2026-09-01`.)

The **Human Disease Ontology (DO, `DOID:…`)** is an older, smaller disease ontology with OMIM xrefs; HOW_IT_WORKS reports that it covers about 59% of our OMIM diseases versus about 99% for MONDO. Licences: MONDO **CC BY 4.0** (attribution required); DO **CC0**. Both licence statements are inside the `.obo` headers in `data/raw/ontology/`.

#### 2.4.5 CTD's MEDIC vocabulary

CTD (next section) uses its own disease vocabulary, **MEDIC** (MErged DIsease voCabulary; Davis et al., 2012): the MeSH disease tree (category C plus some F) with OMIM entries grafted on. Each MEDIC term has a primary `DiseaseID` (`MESH:…` or, for OMIM diseases without a MeSH equivalent, `OMIM:…`) and `AltDiseaseIDs` (other MeSH, OMIM and DOID identifiers merged into it). In the project's copy there are 13,323 MEDIC terms: 11,742 with a MeSH primary ID and 1,581 with an OMIM primary ID. OMIM-primary terms get pseudo tree numbers made by appending the OMIM number to a parent's tree number (e.g. `C20/146850`).

A **granularity mismatch** is common: OMIM `104300` (a specific familial form) appears in MEDIC only as an *alternative ID of the general* `MESH:D000544` *Alzheimer Disease*. Mapping 104300 → D000544 is not "equivalent"; it is "narrower-to-broader". The project accepts this (and unions it with MONDO's more specific `MESH:C536594`) because it uses CTD only to collect *genes* associated with the disease area — a deliberate, documented approximation.

### 2.5 CTD: the Comparative Toxicogenomics Database

#### 2.5.1 What CTD is

CTD (North Carolina State University; launched in 2004) curates, from the published literature, how **chemicals** interact with **genes/proteins**, and how chemicals and genes relate to **diseases** and **phenotypes**. Professional biocurators read PubMed articles and record structured statements, each backed by PubMed IDs. CTD then **integrates** these direct statements to *infer* new relationships. The 2025 update (Davis et al., 2025) reports about 3.8 million manually curated direct interactions from more than 149,000 articles (as of August 2024), integrated into some 48 million inferred relationships. CTD uses controlled vocabularies throughout: MeSH for chemicals (the chemical "vocabulary" is MeSH descriptors and SCRs, IDs like `D008727`), NCBI Gene for genes, and MEDIC for diseases.

#### 2.5.2 Curated versus inferred

This is the single most important distinction in CTD, and the one most relevant to leakage.

- A **curated** (direct-evidence) link was stated in a paper and recorded by a curator: *"methotrexate is used to treat rheumatoid arthritis (PMID …)"*.
- An **inferred** link was *computed* by CTD by chaining two curated links through a shared intermediate:

```
 inferred chemical–disease:   chemical C ──curated──► gene G ──curated──► disease D     ⇒  C ~ D  "inferred via G"
 inferred gene–disease:       gene G ──curated──► chemical C ──curated──► disease D     ⇒  G ~ D  "inferred via C"
```

Inferred chemical–disease rows carry the gene they were inferred through (`InferenceGeneSymbol`) and an `InferenceScore`, a measure of how strongly the local network connects the chemical, the gene(s) and the disease (King et al., 2012). An inference is a *hypothesis generator*, not evidence: it says "these two are connected through the curated network", which is exactly the kind of statement a link-prediction model itself makes.

#### 2.5.3 `DirectEvidence` values

Curated chemical–disease and gene–disease rows carry a `DirectEvidence` value. CTD's glossary defines them as follows (paraphrased):

| Relationship | `marker/mechanism` | `therapeutic` |
|---|---|---|
| chemical–disease | the chemical correlates with the disease (e.g. raised levels) or may play a role in causing it | the chemical has a known or potential therapeutic role in the disease |
| gene–disease | the gene may be a biomarker of the disease or play a role in its etiology | the gene is or may be a therapeutic target in treating the disease |

In the project's curated gene–disease file (34,315 rows) the values are: `marker/mechanism` 32,044; `therapeutic` 1,961; and `marker/mechanism|therapeutic` (both) 310. In the curated chemical–disease rows kept for our drugs (38,859 rows, 614 CTD chemicals) there are 25,029 `marker/mechanism` and 13,830 `therapeutic` rows; there the two kinds of evidence appear as separate rows.

Note what a **therapeutic chemical–disease** link is: a *drug indication* or a drug tested against a disease — the very thing our model predicts. And a **therapeutic gene–disease** link is often recorded *because a drug acting on that gene treats the disease*. Both carry label information (Section 8, question 2).

#### 2.5.4 The files

The project downloads five CTD files (tab-separated, gzip-compressed). Every file starts with a block of `#` comment lines: the copyright notice, the terms of use, a `# Report created:` date stamp, and `# Fields:` followed by the column names. The data rows follow.

| File (size) | One row = | Columns |
|---|---|---|
| `CTD_chemicals.tsv.gz` (10.5 MB) | one CTD chemical (MeSH descriptor or SCR) | ChemicalName, ChemicalID, CasRN, PubChemCID, PubChemSID, DTXSID, InChIKey, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers, MESHSynonyms, CTDCuratedSynonyms |
| `CTD_diseases.tsv.gz` (1.8 MB) | one MEDIC disease | DiseaseName, DiseaseID, AltDiseaseIDs, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers, Synonyms, SlimMappings |
| `CTD_chem_gene_ixns.tsv.gz` (43.3 MB) | one curated chemical–gene interaction statement | ChemicalName, ChemicalID, CasRN, GeneSymbol, GeneID, GeneForms, Organism, OrganismID, Interaction, InteractionActions, PubMedIDs |
| `CTD_curated_genes_diseases.tsv.gz` (0.6 MB) | one curated gene–disease link | GeneSymbol, GeneID, DiseaseName, DiseaseID, DirectEvidence, OmimIDs, PubMedIDs |
| `CTD_chemicals_diseases.tsv.gz` (162 MB) | one chemical–disease link, curated **or** inferred | ChemicalName, ChemicalID, CasRN, DiseaseName, DiseaseID, DirectEvidence, InferenceGeneSymbol, InferenceScore, OmimIDs, PubMedIDs |

Format details that bite (all verified on the project's files, report date 29 September 2026):
- **ID prefixes are inconsistent across files.** `ChemicalID` is `MESH:D008727` in `CTD_chemicals` but bare `D008727` in the interaction files; `DiseaseID` always has a prefix (`MESH:` or `OMIM:`); `PubChemCID` is `CID:2244`, `PubChemSID` is `SID:53788943`.
- **Empty means "not curated", not zero.** An inferred chemical–disease row has an empty `DirectEvidence`; a curated row has empty `InferenceGeneSymbol`/`InferenceScore`.
- **Multi-valued fields** use `|` (synonyms, PubMed IDs, OMIM IDs, tree numbers); interaction actions use `^` inside items (`increases^expression`).
- **The `InChIKey` column of `CTD_chemicals` is empty in this release** (0 of 179,669 rows), and only 11,688 chemicals have a PubChem CID.
- **`#` occurs inside some data rows** (e.g. a synonym `DIM #34`, dye names). A reader configured with "`#` starts a comment" (`pandas.read_csv(comment="#")`) truncates those rows at the `#`. It affects 14 rows of `CTD_chemicals`; harmless for the columns the project needs, but worth knowing.

**Terms of use** (from every file header): cite CTD in any publication; link back to CTD pages in online applications; notify CTD and describe your use; give CTD access to your publication for quality control.

### 2.6 ClinicalTrials.gov

ClinicalTrials.gov (NLM) is the registry of clinical studies conducted around the world; each study has an **NCT number** (`NCT` + 8 digits). For repositioning it is the natural "has anyone tested drug X in disease Y?" check.

**API v2** (the modernized API, OpenAPI 3.0 specification):

```
GET https://clinicaltrials.gov/api/v2/studies?query.cond=Alzheimer+disease&query.intr=donepezil&countTotal=true&pageSize=1
```

| Parameter | Meaning |
|---|---|
| `query.cond` | condition/disease search terms |
| `query.intr` | intervention/treatment search terms |
| `query.term` | free-text search over the whole record |
| `filter.overallStatus` | e.g. `COMPLETED`, `RECRUITING` |
| `fields` | which fields to return (smaller responses) |
| `pageSize` | studies per page (up to 1,000) |
| `pageToken` | continue from the `nextPageToken` of the previous page |
| `countTotal=true` | include `totalCount` in the response |

Other endpoints: `/api/v2/studies/{NCT ID}` (one study), `/api/v2/version` (API version and the `dataTimestamp` of the last data refresh). Data are refreshed daily on weekdays. No API key is needed. ClinicalTrials.gov does not publish an exact rate limit on its API page; community clients assume about 50 requests per minute per IP, so stay well below that and cache results.

**What a count means.** `totalCount` is the number of *registered study records* whose condition and intervention fields match your search terms. It is not "evidence that the drug works": trials can be observational, terminated, combination therapies, or the drug may be a comparator. The search matches terms and some related forms, so the number depends on the exact wording (which is why `06_case_study.py` simplifies disease names before querying). Record the query date with every count: on 1 October 2026, donepezil × "Alzheimer disease" returned 208 studies (Section 4.9).

### 2.7 Connectivity Map and LINCS L1000 (briefly)

The **Connectivity Map** (CMap; Lamb et al., 2006) measured genome-wide expression in human cell lines treated with about 1,300 small molecules, and introduced the idea of *connectivity*: if a disease signature (genes up/down in disease) is **reversed** by a drug's signature, the drug is a repositioning candidate. **LINCS L1000** (Subramanian et al., 2017) scaled this up with a cheap assay that directly measures 978 "landmark" transcripts and infers most of the rest, producing over a million profiles for tens of thousands of perturbagens (compounds, gene knock-downs, over-expressions). Data are available through clue.io (registration) and NCBI GEO. Identifiers are their own (`BRD-…` Broad compound IDs, cell-line names) and must be mapped to DrugBank/PubChem via InChIKey or names — another entity-resolution job. The project lists L1000 as future work (HOW_IT_WORKS section 9): the files are tens of gigabytes, signatures need careful processing (cell line, dose, time point), and only part of our drugs are covered.

### 2.8 Entity resolution: the theory

#### 2.8.1 The problem, stated precisely

Let $X$ be the entities in our dataset (e.g. 682 DrugBank IDs) and $Y$ the records of a target database (e.g. 179,669 CTD chemicals). **Entity resolution** (also called record linkage, identifier mapping) builds a relation

$$M \subseteq X \times Y, \qquad (x, y) \in M \iff \text{"$x$ and $y$ denote the same real-world entity"}.$$

Four numbers describe any mapping:

- **Coverage** $= |\{x : \exists y,\ (x,y)\in M\}| \,/\, |X|$ — the share of our entities that found a partner.
- **Precision** $=$ share of pairs in $M$ that are correct. You cannot compute it automatically; estimate it by **auditing a random sample** by hand (e.g. 50 pairs) and report the result.
- **Cardinality** — for each $x$, how many $y$ (fan-out), and for each $y$, how many $x$ (fan-in). Classes: one-to-one (1:1), one-to-many (1:n), many-to-one (n:1), many-to-many (n:m).
- **Provenance per pair** — *which method* produced it (exact xref, InChIKey, name…), so that you can later filter by reliability.

#### 2.8.2 Why cardinality matters: the join-explosion formula

When you join two tables on a key, each key value $k$ that appears $a_k$ times on the left and $b_k$ times on the right produces $a_k b_k$ output rows. The output has

$$N_{\text{out}} = \sum_k a_k\, b_k \quad\text{(inner join)}, \qquad N_{\text{out}}^{\text{left}} = \sum_k a_k \max(b_k, 1) \quad\text{(left join)}$$

rows. If every key is unique on the right ($b_k \le 1$), a left join keeps exactly the left table's row count. If not, rows multiply silently: a disease that maps to three CUIs becomes three rows, and anything you later sum or average over diseases is now weighted 3×. Section 4.8 shows 415 diseases turning into 628 rows. pandas can *assert* the expected cardinality with `merge(..., validate="one_to_one" | "one_to_many" | "many_to_one")` and raises an error if the data violate it. Use it on every join.

#### 2.8.3 Strategies, from most to least reliable

1. **Authoritative exact cross-reference.** The owner of $x$ itself states the link. *Example:* DrugBank depositing `DB00945` as a PubChem substance. Highest reliability, because the owner knows what it meant.
2. **Third-party curated cross-reference.** Another curator states the link, ideally with a qualifier (MONDO's `equivalentTo`, MedGen's `OMIM` vs `OMIM included`, CTD's `PubChemCID`). Reliable when qualified; check the qualifier and check for many-to-one targets (Section 1: methotrexate's CID → "Folic Acid Antagonists").
3. **Structure key.** For chemicals, compute or fetch the **standard InChIKey** on both sides and join. Full-key match ⇒ same structure (same stereo, same protonation). Block-1 match ⇒ same skeleton, possibly different stereo/isotopes — use only as a flagged, lower-confidence match. Normalize first (strip salts/solvents, neutralize) if you want parent-level matching.
4. **Hub services.** UniChem (chemicals), MONDO / MedGen / UMLS (diseases), HGNC / NCBI Gene (genes), and **Bioregistry** (which knows every prefix, its correct spelling and its URL pattern) do the cross-referencing for you at scale.
5. **Name and synonym matching.** Normalize (case-fold, trim, collapse whitespace, normalize Unicode and Greek letters, remove salt words like "hydrochloride" only if you intend parent-level matching), then match against names *and* synonyms. Then **check ambiguity**: if a normalized name maps to more than one target, do not pick one silently.
6. **Fuzzy matching and manual curation** for the remainder — fuzzy string similarity (edit distance) is fine for *suggesting* candidates to a human, dangerous as an automatic rule (e.g. "estradiol" vs "estriol").

A **cascade** applies these in order, and for each entity records the first method that succeeded. A better cascade also runs the *later* methods for already-mapped entities and **compares**: if two independent routes agree, confidence goes up; if they disagree, the entity goes to a review list. Section 4.7 does exactly this for our drugs and finds 211 agreements and 10 conflicts.

#### 2.8.4 Worked example: a cascade with honest accounting

Using the project's real numbers for the 682 unique drugs:

```
682 DrugBank IDs
 ├─ PubChem (DrugBank's own substance → CID):         667 with CID, SMILES and InChIKey    (97.8%)
 │     15 without: 9 biologics or mixtures (sermorelin, salmon calcitonin, teriparatide, glatiramer,
 │                 heparin, enoxaparin, ardeparin, porfimer sodium, conjugated estrogens),
 │                 3 retired IDs (DB00510, DB01258, DB01402), and 3 small molecules whose
 │                 DrugBank substance returned no compound (propoxyphene, folinic acid, dyphylline)
 └─ CTD chemical:                                     643 mapped                             (94.3%)
       route actually used:  name/synonym = 643,  CID = 0,  InChIKey = 0      (Section 4.7)
       if the CID route is repaired: both routes agree for 211, conflict for 10,
                                     CID-only would add 7, name-only 422, neither 32
```

Note that 12 of the 15 drugs without a compound still map to CTD by name (e.g. heparin → `D006493`), which is why the CTD count can exceed what a structure-only route could reach. A coverage line like this belongs in every methods section.

### 2.9 A catalogue of pitfalls

| Pitfall | What happens | Real example in this project | Defence |
|---|---|---|---|
| **Prefix/format mismatch** | Join matches nothing, silently | CTD `CID:126941` vs PubChem `126941` | Normalize IDs to one form; assert that each join step matches > 0 rows |
| **Empty-key join** | Empty strings (or NaN converted to `""`) match each other | Naive `isin` matched 15 drugs with empty CID to CTD rows with empty CID | Drop empty keys before joining |
| **Namespace collision** | A bare number means different things in different namespaces | `602439` is an OMIM number and a MedGen internal ID ("Voice testing inconclusive") | Keep CURIEs; filter on the source column |
| **Many-to-one target** | Several entities map to one broad record | Methotrexate's CID → "Folic Acid Antagonists" in CTD | Inspect fan-in; check names of targets; prefer exact-name records |
| **One-to-many** | One entity maps to several records; a join multiplies rows | OMIM 104300 has four MedGen rows | Filter by qualifier; `validate=`; aggregate deliberately (e.g. max similarity) |
| **Retired / merged IDs** | ID no longer resolves, or resolves to a merged record | 3 DrugBank IDs; 5 OMIM numbers absent from MedGen whose MONDO xref is `equivalentObsolete` | Track obsolete status; map via secondary IDs; report the loss |
| **Granularity mismatch** | Specific ↔ general concept | OMIM 104300 (familial type 1) ↔ MeSH D000544 (all Alzheimer disease) | Record the relation type; decide if "broader" is acceptable for your use |
| **Salt / stereo / tautomer** | Same drug, different structure strings | metformin vs metformin HCl; omeprazole vs esomeprazole | Parent-normalize; compare full key vs block 1 |
| **Name ambiguity** | One name, several entities | "tcdd" → two CTD records; "heparin" is a family | Detect names with > 1 target; never use "first match wins" silently |
| **Odd source titles** | Bad names defeat name matching | PubChem title `Npc209773` for DrugBank's atropine (`DB00572`): no CTD match | Prefer the depositor's own name (DrugBank synonyms) |
| **Version drift** | A file's format or content changes between downloads | CTD added the `CID:` prefix; its InChIKey column is now empty | Record versions; re-validate coverage after every re-download |
| **Comment characters in data** | Rows truncated by the parser | 14 `CTD_chemicals` rows contain `#` | Skip header lines explicitly instead of `comment="#"` |
| **Case / whitespace / Unicode** | "Aspirin " ≠ "aspirin" | — | Normalize before matching |

### 2.10 Provenance and licensing

**Provenance** is the record of *where each piece of data came from and what was done to it*. A reviewer (or you, in six months) must be able to answer: which file, from which URL, downloaded when, which internal version, unchanged since (checksum), processed by which code, under which licence. Minimal practice:

1. Keep `data/raw/` **immutable**: never edit a downloaded file; derive everything with scripts.
2. Write a **manifest** (JSON/CSV) next to the raw data: URL, download timestamp (UTC), byte size, SHA-256 checksum, the file's own version stamp (`# Report created:` for CTD, `data-version:` for OBO files, DrugBank release number), and licence. Section 4.10 builds one.
3. **Cache API results with the query date** (the project's `pubchem_drugs.csv`; ClinicalTrials.gov counts in the case-study CSVs).
4. In the paper's methods: name every source, its version or download date, and cite its reference paper.

**Licences of the sources in this project:**

| Source | Licence / terms | What it means for us |
|---|---|---|
| DrugBank full data | CC BY-NC 4.0; academic registration | Not downloaded; we use only DrugBank **IDs** (from the benchmarks) |
| DrugBank Open Data (vocabulary, structures) | CC0 | Free to use and share |
| PubChem | Free public NCBI resource; some depositors state their own terms on their source pages | Fine for research; cite PubChem |
| CTD | CTD terms (cite, link, notify, give access) | Cite Davis et al.; notify CTD when publishing |
| OMIM | © Johns Hopkins; registration for bulk files; licence for commercial use or redistribution | We avoid OMIM files; names come from MedGen |
| MedGen | Free NCBI resource; component vocabularies keep their own terms | Cite MedGen |
| UMLS | Free UMLS licence required | Not used directly |
| MONDO | CC BY 4.0 | Attribute MONDO |
| Disease Ontology | CC0 | Free |
| ClinicalTrials.gov | Free public registry (see its terms) | Record query dates |
| ChEMBL | CC BY-SA 3.0 | Share-alike for redistributed derivatives |
| KEGG | Free website for academics; FTP needs a subscription | Not used |
| Benchmarks (Fdataset, Cdataset) | Published supplementary data (Gottlieb 2011; Luo 2016), mirrored by the DRHGCN repository | Cite the original papers and the mirror |

---

## 3. Worked examples by hand

### Worked example 1 — aspirin across nine identifier systems

| System | Identifier | How you would find it |
|---|---|---|
| DrugBank | `DB00945` | the benchmark row |
| PubChem Compound | CID `2244` | `/substance/sourceid/DrugBank/DB00945/cids/TXT` |
| Standard InChIKey | `BSYNRYMUTXBXSQ-UHFFFAOYSA-N` | `/compound/cid/2244/property/InChIKey/TXT` |
| ChEMBL | `CHEMBL25` | UniChem lookup by InChIKey |
| KEGG DRUG | `D00109` | KEGG search |
| MeSH / CTD chemical | `D001241` (`MESH:D001241` in `CTD_chemicals`) | CTD row "Aspirin", which also lists `CID:2244`, `SID:53788943`, `DTXSID5020108`, CAS `50-78-2` |
| Main target gene | NCBI Gene `5742`, HGNC `PTGS1` / `HGNC:9604` | DrugBank target list; HGNC |
| Target protein | UniProt `P23219` | UniProt |

Notice that the *same* concept carries a different ID in every system, and that only some systems link to each other directly. Every arrow in a pipeline is a place where mappings can be lost or corrupted.

### Worked example 2 — reading an OMIM number

`104300`: first digit **1** ⇒ autosomal entry created before 15 May 1994. On omim.org it is shown as **`#104300`** ⇒ a phenotype entry whose molecular basis is known (the genes, e.g. APP, have their own `*` entries). So `104300` is a valid *disease* column. By contrast `*104760` (APP) is a gene entry and would be a category error as a disease. A number shown with `^` would mean the entry was removed or moved; you would then look for the entry it moved to.

### Worked example 3 — row-count arithmetic for a join

Left table: 3 diseases `{A, B, C}`. Right table (concept mappings): `A→c1`, `A→c2`, `B→c3`, `B→c4`, `B→c5`, and no row for `C`.
- Inner join rows $= 1\cdot 2 + 1\cdot 3 + 1\cdot 0 = 5$.
- Left join rows $= 2 + 3 + \max(0,1) = 6$ (C appears once with missing values).
- If you now compute "mean number of genes per disease" over the joined table, B counts three times. Correct approach: decide the cardinality you want (e.g. keep only the equivalent concept, giving 1:1), or aggregate per disease *before* any averaging.

### Worked example 4 — what to do with each kind of CTD row

| Row | Use as model input? | Use for validation? | Why |
|---|---|---|---|
| curated chemical–gene interaction (human) | Yes (`gene_r`) | — | describes the drug's biology, not its indications (with a residual-risk caveat, Section 5) |
| curated gene–disease, `marker/mechanism` | Yes (`gene_d`) | — | describes disease biology |
| curated gene–disease, `therapeutic` only | **No** | — | encodes "a drug targeting this gene treats this disease" |
| inferred gene–disease | **No** | — | built by chaining through chemical–disease links, many of them therapeutic ⇒ label information |
| curated chemical–disease, `therapeutic` | **No** | Yes (case studies) | this *is* an indication label |
| curated chemical–disease, `marker/mechanism` | No | Yes, weaker | often adverse effects/toxicity rather than treatment |
| inferred chemical–disease | No | No | a computed hypothesis, not independent evidence |

### Worked example 5 — deciding a mapping by hand

The drug "Heparin Pentasaccharide" (`DB00569`, CID 5282448). The CID route (after removing the `CID:` prefix) finds CTD record `C064764` "PENTA"; the name route finds `C049714` "IC 831423" (via a synonym). The two routes disagree and both target names are cryptic codes. Correct action: **do not** accept either automatically; put the drug on a review list, look both records up on ctdbase.org, and decide (or leave it unmapped and say so). In the project's current output this drug is mapped by name to `C049714` and has zero human gene interactions, so the decision has no effect on the features — but you only know that because you checked.

---

## 4. Code: working with the real files

All blocks run from the project root with `.venv\Scripts\python.exe`. They only read files; none of them modifies anything in the project. Run them in order (each is self-contained).

### 4.1 Provenance first: read a CTD header without reading the file

Before using any file, find out *what version it is* and *what its columns are*. CTD puts both in the `#` header. Reading stops at the first data row, so even the 162 MB file costs only a few milliseconds.

```python
import gzip
from pathlib import Path

def ctd_header(path):
    """Read only the '#' comment block at the top of a CTD file: report date + column names."""
    created, fields = None, None
    with gzip.open(path, "rt", encoding="utf8") as fh:
        prev = ""
        for line in fh:
            if not line.startswith("#"):
                break                                   # first data row: stop reading
            text = line[1:].strip()
            if text.startswith("Report created:"):
                created = text.split(":", 1)[1].strip()
            if prev == "Fields:":
                fields = text.split("\t")
            prev = text
    return created, fields

for p in sorted(Path("data/raw/ctd").glob("*.tsv.gz")):
    created, fields = ctd_header(p)
    print(f"{p.name:38s} {p.stat().st_size / 1e6:6.1f} MB  created {created}")
    print(f"    {len(fields)} fields: {', '.join(fields)}")
```

Output:

```text
CTD_chem_gene_ixns.tsv.gz                43.3 MB  created Tue Sep 29 13:22:16 EDT 2026
    11 fields: ChemicalName, ChemicalID, CasRN, GeneSymbol, GeneID, GeneForms, Organism, OrganismID, Interaction, InteractionActions, PubMedIDs
CTD_chemicals.tsv.gz                     10.5 MB  created Tue Sep 29 13:07:01 EDT 2026
    13 fields: ChemicalName, ChemicalID, CasRN, PubChemCID, PubChemSID, DTXSID, InChIKey, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers, MESHSynonyms, CTDCuratedSynonyms
CTD_chemicals_diseases.tsv.gz           162.3 MB  created Tue Sep 29 13:17:30 EDT 2026
    10 fields: ChemicalName, ChemicalID, CasRN, DiseaseName, DiseaseID, DirectEvidence, InferenceGeneSymbol, InferenceScore, OmimIDs, PubMedIDs
CTD_curated_genes_diseases.tsv.gz         0.6 MB  created Tue Sep 29 13:24:49 EDT 2026
    7 fields: GeneSymbol, GeneID, DiseaseName, DiseaseID, DirectEvidence, OmimIDs, PubMedIDs
CTD_diseases.tsv.gz                       1.8 MB  created Tue Sep 29 13:07:12 EDT 2026
    9 fields: DiseaseName, DiseaseID, AltDiseaseIDs, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers, Synonyms, SlimMappings
```

Things to notice: the **report date** is the file's real version (CTD has no release numbers); the **inconsistent field sets** across files are why each reader in `02_build_features.py` passes its own `names=` list.

### 4.2 Curated versus inferred rows, without loading 162 MB

`CTD_chemicals_diseases.tsv.gz` decompresses to well over a gigabyte. To look at its structure we stream lines lazily (`gzip.open` + a generator + `itertools.islice`), so memory use stays tiny.

```python
import gzip
from collections import Counter
from itertools import islice

COLS = ["ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID", "DirectEvidence",
        "InferenceGeneSymbol", "InferenceScore", "OmimIDs", "PubMedIDs"]

def stream_rows(path, n):
    """Yield the first n DATA rows as dicts, without ever loading the whole file."""
    with gzip.open(path, "rt", encoding="utf8") as fh:
        data = (l for l in fh if not l.startswith("#"))
        for line in islice(data, n):
            yield dict(zip(COLS, line.rstrip("\n").split("\t")))

rows = list(stream_rows("data/raw/ctd/CTD_chemicals_diseases.tsv.gz", 5000))
kind = Counter("curated: " + r["DirectEvidence"] if r["DirectEvidence"] else "inferred"
               for r in rows)
print("first 5000 rows:", dict(kind))
cur = next(r for r in rows if r["DirectEvidence"] == "therapeutic")
inf = next(r for r in rows if not r["DirectEvidence"])
for label, r in [("CURATED", cur), ("INFERRED", inf)]:
    print(f"\n{label}")
    for c in COLS:
        print(f"  {c:20s} {r[c]!r}")
```

Output:

```text
first 5000 rows: {'curated: therapeutic': 9, 'inferred': 4984, 'curated: marker/mechanism': 7}

CURATED
  ChemicalName         '06-Paris-LA-66 protocol'
  ChemicalID           'C046983'
  CasRN                ''
  DiseaseName          'Precursor Cell Lymphoblastic Leukemia-Lymphoma'
  DiseaseID            'MESH:D054198'
  DirectEvidence       'therapeutic'
  InferenceGeneSymbol  ''
  InferenceScore       ''
  OmimIDs              ''
  PubMedIDs            '4519131'

INFERRED
  ChemicalName         '10074-G5'
  ChemicalID           'C534883'
  CasRN                ''
  DiseaseName          'Adenocarcinoma'
  DiseaseID            'MESH:D000230'
  DirectEvidence       ''
  InferenceGeneSymbol  'MYC'
  InferenceScore       '4.08'
  OmimIDs              ''
  PubMedIDs            '26432044'
```

Among the first 5,000 rows (chemicals sorted alphabetically), 99.7% are **inferred**. That ratio is typical: inferred links vastly outnumber curated ones. The inferred row says only that `10074-G5` interacts with MYC and MYC is curated as associated with adenocarcinoma; no paper says `10074-G5` affects adenocarcinoma. Also notice the ID formats: `ChemicalID` has no prefix here, `DiseaseID` does.

### 4.3 MEDIC: from an OMIM number to a CTD disease

```python
import pandas as pd

COLS = ["DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition", "ParentIDs",
        "TreeNumbers", "ParentTreeNumbers", "Synonyms", "SlimMappings"]
medic = pd.read_csv("data/raw/ctd/CTD_diseases.tsv.gz", sep="\t", comment="#",
                    names=COLS, dtype=str).fillna("")
print(f"MEDIC terms: {len(medic)}")
print("primary ID namespaces:", medic.DiseaseID.str.split(":").str[0].value_counts().to_dict())

# Build OMIM -> MEDIC index from BOTH the primary ID and the alternative IDs
omim2medic = {}
for r in medic.itertuples():
    for i in [r.DiseaseID] + r.AltDiseaseIDs.split("|"):
        if i.startswith("OMIM:"):
            omim2medic.setdefault(i[5:], []).append(r.DiseaseID)
print(f"OMIM numbers reachable in MEDIC: {len(omim2medic)}")

for omim in ["104300", "146850", "144400"]:
    hits = omim2medic.get(omim, [])
    print(f"\nOMIM {omim} -> {hits or 'no MEDIC term'}")
    for h in hits:
        t = medic.set_index("DiseaseID").loc[h]
        print(f"   {h}  '{t.DiseaseName}'")
        print(f"   tree numbers: {t.TreeNumbers}")
        print(f"   depth of each path: {[len(x.split('/')[0].split('.')) for x in t.TreeNumbers.split('|')]}")
```

Output:

```text
MEDIC terms: 13323
primary ID namespaces: {'MESH': 11742, 'OMIM': 1581}
OMIM numbers reachable in MEDIC: 5618

OMIM 104300 -> ['MESH:D000544']
   MESH:D000544  'Alzheimer Disease'
   tree numbers: C10.228.140.380.100|C10.574.945.249|F03.615.400.100
   depth of each path: [5, 4, 4]

OMIM 146850 -> ['OMIM:146850']
   OMIM:146850  'IMMUNE SUPPRESSION'
   tree numbers: C01.150.252.410.890/146850|C20/146850
   depth of each path: [5, 1]

OMIM 144400 -> no MEDIC term
```

Three different outcomes for three OMIM numbers: `104300` maps (as an alternative ID) to the broader MeSH descriptor *Alzheimer Disease*; `146850` has its own OMIM-primary MEDIC term with pseudo tree numbers (`…/146850`); `144400` is not in MEDIC at all. Only 5,618 OMIM numbers are reachable through MEDIC, far fewer than OMIM's phenotype entries, which is why the project unions MEDIC with MeSH IDs obtained through MONDO and MedGen.

### 4.4 MedGen: one OMIM number, four concepts

```python
import pandas as pd

mg = pd.read_csv("data/raw/medgen/MedGenIDMappings.txt.gz", sep="|", dtype=str, usecols=[0, 1, 2, 3])
mg.columns = ["cui", "name", "src_id", "src"]
print(f"MedGen mapping rows: {len(mg):,}")
print("top sources:", mg.src.value_counts().head(8).to_dict())

print("\nEvery row that mentions OMIM 104300:")
print(mg[mg.src_id == "104300"].to_string(index=False))

ours = pd.read_csv("data/interim/diseases_all.csv", dtype=str).omim       # our 415 OMIM IDs
om = mg[(mg.src == "OMIM") & mg.src_id.isin(ours)]
per_omim = om.groupby("src_id").cui.nunique()
per_cui = mg[mg.src == "OMIM"].groupby("cui").src_id.nunique()
print(f"\nour OMIM IDs found with src == 'OMIM': {per_omim.size} of {ours.size}")
print("CUIs per OMIM ID (ours):", per_omim.value_counts().sort_index().to_dict())
print("OMIM IDs per CUI (all of MedGen):", per_cui.value_counts().sort_index().head(5).to_dict())
dis = pd.read_csv("data/interim/diseases_all.csv", dtype=str).fillna("")
not_mg = dis[~dis.omim.isin(per_omim.index)]
print("\nour diseases with no MedGen row of source OMIM:")
print(not_mg[["omim", "name", "mondo"]].to_string(index=False))
```

Output:

```text
MedGen mapping rows: 421,954
top sources: {'MedGen': 197244, 'SNOMEDCT_US': 123024, 'MONDO': 23318, 'HPO': 19643, 'GARD': 16842, 'MeSH': 16503, 'OMIM': 10249, 'Orphanet': 9026}

Every row that mentions OMIM 104300:
     cui                                                             name src_id           src
C0002395                                                Alzheimer disease 104300 OMIM included
C1863053 Alzheimer disease, early-onset, with cerebral amyloid angiopathy 104300 OMIM included
C1863052                                         Alzheimer disease type 1 104300          OMIM
C3549448                            Alzheimer disease, protection against 104300 OMIM included

our OMIM IDs found with src == 'OMIM': 405 of 415
CUIs per OMIM ID (ours): {1: 405}
OMIM IDs per CUI (all of MedGen): {1: 10233, 2: 8}

our diseases with no MedGen row of source OMIM:
  omim                                                                          name         mondo
144400                                                                                            
175505                  gastric adenocarcinoma and proximal polyposis of the stomach MONDO:0017790
212110                                                                                            
259660                                          lethal osteosclerotic bone dysplasia MONDO:0009821
300494                             Asperger syndrome, X-linked, susceptibility to, 1 MONDO:0010340
305300                                                                                            
600634                                   prolactin-producing pituitary gland adenoma MONDO:0010911
602439                                                                                            
605839                                                                                            
608088 cerebellar ataxia with neuropathy and bilateral vestibular areflexia syndrome MONDO:0044720
```

Read the 104300 block carefully: four CUIs mention OMIM 104300, but only **one** row has source `OMIM` (the equivalent concept, "Alzheimer disease type 1"); the other three are `OMIM included` — the general disease, a variant form, and a *protective* allele. A join that ignores the source column would give 104300 four names, one of which means the opposite of the disease. After filtering on `src == "OMIM"`, our 405 found diseases are exactly 1:1 (one CUI each). Ten diseases have no `OMIM` row in MedGen; the next block explains five of them.

### 4.5 MONDO: cross-references and their qualifiers

```python
import re
from collections import Counter, defaultdict
import pandas as pd

def parse_obo_xrefs(path):
    """Return {term id: {'name', 'obsolete', 'xrefs': [(xref, qualifier), ...]}}."""
    terms, cur = {}, None
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("["):
                cur = {"xrefs": [], "obsolete": False, "name": ""} if line == "[Term]" else None
            elif cur is not None:
                if line.startswith("id: "):
                    terms[line[4:]] = cur
                elif line.startswith("name: "):
                    cur["name"] = line[6:]
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
                elif line.startswith("xref: "):
                    xref = line[6:].split(" ")[0]
                    m = re.search(r'source="MONDO:(\w+)"', line)      # the mapping qualifier
                    cur["xrefs"].append((xref, m.group(1) if m else "unqualified"))
    return terms

path = "data/raw/ontology/mondo.obo"
with open(path, encoding="utf8") as fh:
    version = next(l for l in fh if l.startswith("data-version")).strip()
terms = parse_obo_xrefs(path)
live = {k: v for k, v in terms.items() if k.startswith("MONDO:") and not v["obsolete"]}
print(version, f"| live MONDO terms: {len(live)}")

q = Counter(qual for v in live.values() for x, qual in v["xrefs"] if x.startswith("OMIM:"))
print("OMIM xrefs on live terms, by qualifier:", dict(q))

omim2mondo = defaultdict(set)
for k, v in live.items():
    for x, qual in v["xrefs"]:
        if x.startswith("OMIM:"):
            omim2mondo[x[5:]].add((k, qual))
print(f"OMIM IDs with more than one live MONDO term: "
      f"{sum(len(s) > 1 for s in omim2mondo.values())} of {len(omim2mondo)}")
odd = sorted((o, m, live[m]["name"], qual) for o, s in omim2mondo.items()
             for m, qual in s if qual != "equivalentTo")
print("first non-equivalentTo OMIM xref:", odd[0])

ours = set(pd.read_csv("data/interim/diseases_all.csv", dtype=str).omim)
print("our OMIM IDs among them:", sorted(o for o, *_ in odd if o in ours))
n_ps = sum(x.startswith("OMIMPS:") for v in live.values() for x, _ in v["xrefs"])
print(f"OMIMPS: (phenotypic-series) xrefs, a DIFFERENT namespace: {n_ps}")
print("MONDO:0007088 xrefs:", live["MONDO:0007088"]["xrefs"])
```

Output:

```text
data-version: releases/2026-09-01 | live MONDO terms: 32109
OMIM xrefs on live terms, by qualifier: {'equivalentTo': 9773, 'equivalentObsolete': 131}
OMIM IDs with more than one live MONDO term: 0 of 9904
first non-equivalentTo OMIM xref: ('102730', 'MONDO:0020458', 'hemolytic anemia due to erythrocyte adenosine deaminase overproduction', 'equivalentObsolete')
our OMIM IDs among them: ['175505', '259660', '300494', '600634', '608088']
OMIMPS: (phenotypic-series) xrefs, a DIFFERENT namespace: 614
MONDO:0007088 xrefs: [('DECIPHER:48', 'relatedTo'), ('DOID:0080348', 'equivalentTo'), ('GARD:0009465', 'GARD'), ('MEDGEN:354892', 'equivalentTo'), ('MESH:C536594', 'equivalentTo'), ('OMIM:104300', 'equivalentTo'), ('UMLS:C1863052', 'equivalentTo')]
```

Two findings. First, within live MONDO terms the OMIM cross-references are one-to-one (no OMIM number is claimed by two live terms), so `parse_mondo`'s list per OMIM ID has length 1 for every disease in this release. Second, **131** OMIM xrefs carry the qualifier `equivalentObsolete`: MONDO keeps the disease but records that the *OMIM entry* has been obsoleted (retired or moved, the `^` category on omim.org). Exactly the five of our diseases that MedGen could not name (175505, 259660, 300494, 600634, 608088) are among them: their 2011-era OMIM entries no longer exist as such, and the project falls back to MONDO's term names. The other five unnamed diseases (144400, 212110, 305300, 602439, 605839) are absent from MedGen, MONDO and the Disease Ontology alike; they are most likely entries OMIM has since removed or merged (omim.org blocks automated look-ups, so check them by hand in a browser).

### 4.6 InChIKey anatomy, salts and stereoisomers

This block uses RDKit (installed in the project environment; see Unit D2).

```python
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

# 1) Anatomy of an InChIKey: 14 chars skeleton - 8 chars stereo/isotope + flag + version - protonation
ik = Chem.MolToInchiKey(Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"))     # aspirin
block1, block2, block3 = ik.split("-")
print(f"aspirin InChIKey {ik}")
print(f"  connectivity block  {block1}  (hash of the molecular skeleton)")
print(f"  stereo/isotope block {block2[:8]} + flag '{block2[8]}' (S = standard) + version '{block2[9]}'")
print(f"  protonation block   {block3}  (N = neutral)")

# 2) A salt and its parent have DIFFERENT keys until you strip the counter-ion
for name, smi in [("metformin", "CN(C)C(=N)N=C(N)N"),
                  ("metformin hydrochloride", "CN(C)C(=N)N=C(N)N.Cl")]:
    mol = Chem.MolFromSmiles(smi)
    parent = rdMolStandardize.FragmentParent(mol)          # keep the largest organic fragment
    print(f"{name:24s} raw {Chem.MolToInchiKey(mol)}   parent {Chem.MolToInchiKey(parent)}")

# 3) Stereoisomers share block 1: find such pairs among OUR drugs
d = pd.read_csv("data/interim/drugs_all.csv", dtype=str).fillna("")
d = d[d.inchikey != ""].assign(skeleton=lambda x: x.inchikey.str[:14])
groups = d.groupby("skeleton").filter(lambda g: len(g) > 1).groupby("skeleton")
print(f"\n{d.inchikey.nunique()} distinct full keys; {groups.ngroups} skeletons shared by >1 drug:")
for sk, g in groups:
    print("  " + " | ".join(f"{r.name} ({r.drugbank_id}, ...{r.inchikey[15:23]})"
                            for r in g.itertuples()))
```

Output:

```text
aspirin InChIKey BSYNRYMUTXBXSQ-UHFFFAOYSA-N
  connectivity block  BSYNRYMUTXBXSQ  (hash of the molecular skeleton)
  stereo/isotope block UHFFFAOY + flag 'S' (S = standard) + version 'A'
  protonation block   N  (N = neutral)
metformin                raw XZWYZXLIPXDOLR-UHFFFAOYSA-N   parent XZWYZXLIPXDOLR-UHFFFAOYSA-N
metformin hydrochloride  raw OETHQSJEHLVLGH-UHFFFAOYSA-N   parent XZWYZXLIPXDOLR-UHFFFAOYSA-N

667 distinct full keys; 12 skeletons shared by >1 drug:
  Epirubicin (DB00445, ...VTZDEGQI) | Doxorubicin (DB00997, ...TZSSRYML)
  Levofloxacin (DB01137, ...JTQLQIEI) | Ofloxacin (DB01165, ...UHFFFAOY)
  Pseudoephedrine (DB00852, ...WCBMZHEX) | Ephedrine (DB01364, ...WPRPVWTQ)
  Amphetamine (DB00182, ...UHFFFAOY) | Dextroamphetamine (DB01576, ...QMMMGPOB)
  Bupivacaine (USAN:INN:BAN) (DB00297, ...UHFFFAOY) | Levobupivacaine (DB01002, ...INIZCTEO)
  Quinine (DB00468, ...WZBLMQSH) | Quinidine (DB00908, ...LHHVKLHA)
  Hyoscyamine (DB00424, ...VFSICIBP) | Npc209773 (DB00572, ...PJPHBNEV)
  9-Cis-Retinoic Acid (DB00523, ...ZVCIMWCZ) | Retinoic Acid (DB00755, ...YCNIQYBT) | Isotretinoin (DB00982, ...XFYACQKR)
  Omeprazole (DB00338, ...UHFFFAOY) | Esomeprazole (DB00736, ...DEOSSOPV)
  Betamethasone (DB00443, ...DVTGEIKX) | Dexamethasone (DB01234, ...CXSFZGCW)
  Citalopram (DB00215, ...UHFFFAOY) | Escitalopram (DB01175, ...FQEVSTJZ)
  Dexbrompheniramine (DB00405, ...HNNXBMFY) | Brompheniramine (DB00835, ...UHFFFAOY)
```

Lessons: (1) the salt and its parent differ until you strip the counter-ion, after which their keys are identical; (2) twelve groups of our drugs share a skeleton — enantiomers (esomeprazole/omeprazole, escitalopram/citalopram, levofloxacin/ofloxacin, dextroamphetamine/amphetamine, levobupivacaine/bupivacaine, dexbrompheniramine/brompheniramine), diastereomers (quinine/quinidine, ephedrine/pseudoephedrine, doxorubicin/epirubicin, dexamethasone/betamethasone) and geometric isomers (the three retinoic acids). They are *different drugs* with different indications, so joining on block 1 alone would be wrong. They are also near-duplicates in the drug–drug similarity views, a fact Unit E1 returns to. (3) `Npc209773` is PubChem's title for DrugBank's atropine (`DB00572`) — a reminder that names from automated titles can be useless for name matching.

### 4.7 Building and auditing the drug → CTD mapping

This is the most important block of the chapter. It rebuilds the project's drug → CTD chemical mapping, but keeps *every* candidate from *each* route and classifies the outcome.

```python
import pandas as pd
from collections import defaultdict

COLS = ["ChemicalName", "ChemicalID", "CasRN", "PubChemCID", "PubChemSID", "DTXSID", "InChIKey",
        "Definition", "ParentIDs", "TreeNumbers", "ParentTreeNumbers", "MESHSynonyms",
        "CTDCuratedSynonyms"]
ch = pd.read_csv("data/raw/ctd/CTD_chemicals.tsv.gz", sep="\t", comment="#", names=COLS,
                 dtype=str, usecols=["ChemicalName", "ChemicalID", "PubChemCID", "InChIKey",
                                     "MESHSynonyms"]).fillna("")
drugs = pd.read_csv("data/interim/drugs_all.csv", dtype=str).fillna("")
print("CTD PubChemCID column looks like:", ch.PubChemCID[ch.PubChemCID != ""].head(3).tolist())
print("CTD rows with an InChIKey:", (ch.InChIKey != "").sum(), "of", len(ch))

# --- naive exact join on the raw strings -----------------------------------
raw = drugs.cid.isin(set(ch.PubChemCID))
print(f"naive join: {raw.sum()} 'matches', of which {(raw & (drugs.cid == '')).sum()} are "
      f"empty string == empty string")
raw_hits = (raw & (drugs.cid != "")).sum()
# --- normalise: strip the 'CID:' and 'MESH:' prefixes before joining ---------
ch["cid"] = ch.PubChemCID.str.removeprefix("CID:")
ch["mesh"] = ch.ChemicalID.str.removeprefix("MESH:")
norm_hits = drugs.cid.isin(set(ch.cid[ch.cid != ""])).sum()
print(f"drugs whose CID matches CTD: raw {raw_hits}, after stripping 'CID:' {norm_hits}")

# --- keep ALL candidates per key (dict of sets), never silently the first ---
by_cid, by_name = defaultdict(set), defaultdict(set)
for r in ch.itertuples():
    if r.cid:
        by_cid[r.cid].add(r.mesh)
    for n in [r.ChemicalName] + r.MESHSynonyms.split("|"):
        if n:
            by_name[n.lower()].add(r.mesh)
name_of = dict(zip(ch.mesh, ch.ChemicalName))

rows = []
for r in drugs.itertuples():
    c, n = by_cid.get(r.cid, set()), by_name.get(r.name.lower(), set())
    status = ("none" if not (c or n) else "cid_only" if not n else "name_only" if not c
              else "agree" if c & n else "CONFLICT")
    rows.append({"drugbank_id": r.drugbank_id, "drug": r.name, "status": status,
                 "n_cid_cands": len(c), "via_cid": "|".join(sorted(c)),
                 "via_name": "|".join(sorted(n))})
m = pd.DataFrame(rows)
print("\nmapping status:", m.status.value_counts().to_dict())
print("drugs whose CID points to >1 CTD record:", (m.n_cid_cands > 1).sum())
print("\nconflicts (CID route vs name route):")
for r in m[m.status == "CONFLICT"].head(6).itertuples():
    cid_names = ", ".join(name_of[x] for x in r.via_cid.split("|"))
    print(f"  {r.drug:22s} CID -> {cid_names!r:48s} name -> {name_of[r.via_name.split('|')[0]]!r}")
```

Output:

```text
CTD PubChemCID column looks like: ['CID:2836600', 'CID:656732', 'CID:83852']
CTD rows with an InChIKey: 0 of 179669
naive join: 15 'matches', of which 15 are empty string == empty string
drugs whose CID matches CTD: raw 0, after stripping 'CID:' 228

mapping status: {'name_only': 422, 'agree': 211, 'none': 32, 'CONFLICT': 10, 'cid_only': 7}
drugs whose CID points to >1 CTD record: 8

conflicts (CID route vs name route):
  Adenosine Monophosphate CID -> 'Adenine Nucleotides'                            name -> 'Adenosine Monophosphate'
  Calcitriol             CID -> '1,25-dihydroxyvitamin D, dihydroxy-vitamin D3'  name -> 'Calcitriol'
  Lidocaine              CID -> 'Lidoderm'                                       name -> 'Lidocaine'
  Rosiglitazone          CID -> 'rosiglitazone-metformin combination'            name -> 'Rosiglitazone'
  Methotrexate           CID -> 'Folic Acid Antagonists'                         name -> 'Methotrexate'
  Heparin Pentasaccharide CID -> 'PENTA'                                          name -> 'IC 831423'
```

What this tells us:

1. **The raw CID join matches nothing.** The 15 "matches" of the naive join are artefacts (empty string equals empty string). After stripping the `CID:` prefix, 228 drugs match by CID.
2. **CTD has no InChIKeys** in this release, so an InChIKey route cannot work at all.
3. **The two routes are independent evidence.** Where both produce a candidate (221 drugs), they agree for 211 and conflict for 10. In the conflicts shown, the *name* route is right and the CID route points to a class ("Folic Acid Antagonists"), a product ("Lidoderm"), a combination ("rosiglitazone-metformin combination") or a related metabolite group ("Adenine Nucleotides"). Exact-ID matching is not automatically better than name matching; cross-references can be attached to the wrong record.
4. **Fan-out exists:** 8 drugs' CIDs point to more than one CTD record (e.g. acetaminophen → *Acetaminophen* and *acetaminophen, hydrocodone drug combination*). A plain `dict` built with `{c: i for c, i in …}` would silently keep the *last* one.
5. A repaired CID route would add 7 mappings that names miss (`cid_only`), e.g. ethinylestradiol and alendronic acid.

A principled resolver would therefore: take `agree` pairs; take `name_only` and `cid_only` pairs but flag them as single-route; and send every `CONFLICT` and every multi-candidate case to a short manual-review list (17 distinct drugs here: 10 conflicts plus 8 multi-candidate CIDs, with calcitriol in both groups). That is a few minutes of work and it makes the mapping defensible.

### 4.8 Many-to-many joins and how to catch them

```python
import pandas as pd

dis = pd.read_csv("data/interim/diseases_all.csv", dtype=str).fillna("")[["omim"]]
mg = pd.read_csv("data/raw/medgen/MedGenIDMappings.txt.gz", sep="|", dtype=str, usecols=[0, 1, 2, 3])
mg.columns = ["cui", "name", "src_id", "src"]

# Careless join: match on the number only, ignoring which namespace it belongs to
careless = dis.merge(mg, left_on="omim", right_on="src_id", how="left")
print(f"{len(dis)} diseases -> {len(careless)} rows after the careless join")
print("sources that matched:", careless.src.value_counts().to_dict())
print(careless[careless.omim.isin(["602439", "605839"])][["omim", "cui", "name", "src"]]
      .to_string(index=False))

# Cardinality report: how many partners does each key have on the other side?
def cardinality(df, left, right):
    l = df.groupby(left)[right].nunique()
    r = df.groupby(right)[left].nunique()
    return {"left keys": len(l), "left keys with >1 partner": int((l > 1).sum()),
            "right keys with >1 partner": int((r > 1).sum())}
print("\ncareless join   :", cardinality(careless.dropna(subset=["cui"]), "omim", "cui"))

# Careful join: right namespace only, and ASSERT the expected cardinality
mg_omim = mg[mg.src == "OMIM"]
careful = dis.merge(mg_omim, left_on="omim", right_on="src_id", how="left",
                    validate="many_to_one")       # raises MergeError if an OMIM had 2 CUIs
print("careful join    :", len(careful), "rows;",
      cardinality(careful.dropna(subset=["cui"]), "omim", "cui"))
try:
    dis.merge(mg, left_on="omim", right_on="src_id", validate="many_to_one")
except pd.errors.MergeError as e:
    print("validate caught the careless join ->", str(e)[:60], "...")
```

Output:

```text
415 diseases -> 628 rows after the careless join
sources that matched: {'OMIM': 405, 'OMIM included': 148, 'MedGen': 68}
  omim      cui                           name    src
602439 C0423543     Voice testing inconclusive MedGen
605839 C0427714 Amniotic fetal cell study: NAD MedGen

careless join   : {'left keys': 408, 'left keys with >1 partner': 128, 'right keys with >1 partner': 0}
careful join    : 415 rows; {'left keys': 405, 'left keys with >1 partner': 0, 'right keys with >1 partner': 0}
validate caught the careless join -> Merge keys are not unique in right dataset; not a many-to-on ...
```

The careless join turned 415 diseases into 628 rows: 148 extra rows from `OMIM included` concepts and 68 rows from MedGen's *internal* numeric IDs that happen to equal one of our OMIM numbers (namespace collision, e.g. 602439 → "Voice testing inconclusive"). Filtering on the namespace restores one row per disease, and `validate="many_to_one"` turns any future violation into an error instead of a silent duplication.

### 4.9 Calling PubChem and ClinicalTrials.gov (two requests only)

These functions show the correct, polite way to call the two web APIs the project uses. The block sends **exactly two requests**. Do not put these calls in a loop without a delay, a cache and an error handler (see `02_build_features.py::_get` for the project's version with retries).

```python
import time
import requests

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"User-Agent": "drug-repositioning-course/1.0 (student project)"}

def pubchem_cids_for_drugbank(db_id):
    """DrugBank ID -> PubChem CID(s), via the SUBSTANCE that DrugBank deposited."""
    url = f"{PUG}/substance/sourceid/DrugBank/{db_id}/cids/TXT"
    r = requests.get(url, headers=HEADERS, timeout=30)
    print("GET", url, "->", r.status_code, "|", r.headers.get("X-Throttling-Control", "")[:60])
    return r.text.split() if r.ok else []

def trial_count(condition, intervention):
    """Number of ClinicalTrials.gov studies matching a condition AND an intervention."""
    r = requests.get("https://clinicaltrials.gov/api/v2/studies", headers=HEADERS, timeout=30,
                     params={"query.cond": condition, "query.intr": intervention,
                             "countTotal": "true", "pageSize": 1})
    print("GET", r.url, "->", r.status_code)
    return r.json().get("totalCount")

print("aspirin CIDs:", pubchem_cids_for_drugbank("DB00945"))
time.sleep(1)                      # be polite: never fire requests back-to-back in a loop
print("donepezil x Alzheimer disease trials:", trial_count("Alzheimer disease", "donepezil"))
```

Output (run on 1 October 2026; the trial count will change over time):

```text
GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/DB00945/cids/TXT -> 200 | Request Count status: Green (1%), Request Time status: Green
aspirin CIDs: ['2244']
GET https://clinicaltrials.gov/api/v2/studies?query.cond=Alzheimer+disease&query.intr=donepezil&countTotal=true&pageSize=1 -> 200
donepezil x Alzheimer disease trials: 208
```

The PubChem response header confirms we are far below the throttling limits. For many drugs, the right pattern is: one request per DrugBank ID for the CID (unavoidable, because the input is a source ID), at most 5 per second, then **one** batched property request per 100 CIDs.

### 4.10 A provenance manifest

```python
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

LICENCES = {   # what you are allowed to do with each file - write it down when you download
    "ctd": "CTD terms of use (cite CTD, link back, notify CTD) - ctdbase.org/about/legal.jsp",
    "medgen": "NCBI public data; cite MedGen; component sources keep their own terms",
    "ontology": "MONDO: CC BY 4.0; Human Disease Ontology: CC0 1.0",
    "benchmarks": "academic benchmark (Gottlieb 2011 / Luo 2016) via github.com/TheWall9/DRHGCN",
}

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()

def internal_version(path):
    """Pull the version stamp that the file itself carries, if any."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith(("# Report created:", "data-version:")):
                return line.strip("# \n")
            if i > 40:
                return None

manifest = []
for p in sorted(Path("data/raw").rglob("*")):
    if p.is_file() and p.suffix in {".gz", ".obo", ".mat"}:
        manifest.append({
            "file": p.as_posix(), "bytes": p.stat().st_size,
            "sha256": sha256(p)[:16] + "...",
            "local_mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"),
            "internal_version": internal_version(p) if p.suffix != ".mat" else None,
            "licence": LICENCES[p.parent.name],
        })
print(json.dumps(manifest[:3], indent=1))
print(f"... {len(manifest)} files in total")
```

Output (first three entries):

```text
[
 {
  "file": "data/raw/benchmarks/Cdataset.mat",
  "bytes": 3470198,
  "sha256": "0d7b715905744784...",
  "local_mtime": "2026-10-01T03:33",
  "internal_version": null,
  "licence": "academic benchmark (Gottlieb 2011 / Luo 2016) via github.com/TheWall9/DRHGCN"
 },
 {
  "file": "data/raw/benchmarks/Fdataset.mat",
  "bytes": 2735480,
  "sha256": "898804861457e1f2...",
  "local_mtime": "2026-10-01T03:33",
  "internal_version": null,
  "licence": "academic benchmark (Gottlieb 2011 / Luo 2016) via github.com/TheWall9/DRHGCN"
 },
 {
  "file": "data/raw/ctd/CTD_chem_gene_ixns.tsv.gz",
  "bytes": 43260596,
  "sha256": "0c069872aeebfd78...",
  "local_mtime": "2026-10-01T03:35",
  "internal_version": "Report created: Tue Sep 29 13:22:16 EDT 2026",
  "licence": "CTD terms of use (cite CTD, link back, notify CTD) - ctdbase.org/about/legal.jsp"
 }
]
... 10 files in total
```

Save this list as `data/raw/MANIFEST.json` in your own copy of the pipeline (the project does not have one yet). A checksum lets anyone verify that their `CTD_chemicals.tsv.gz` is byte-identical to yours; the internal version stamp tells them which CTD release it was even if the file is renamed.

---

## 5. In this project

`02_build_features.py` describes itself as the place where "the whole file is entity resolution". Here is every mapping decision, in pipeline order, with what it gets right and what you should know about it.

### 5.1 `scripts/01_download_data.py` — fetching the raw files

```python
DRHGCN = "https://raw.githubusercontent.com/TheWall9/DRHGCN/master/dataset/"
CTD = "https://ctdbase.org/reports/"

FILES = {
    "benchmarks/Fdataset.mat": DRHGCN + "Fdataset.mat",
    ...
    "ctd/CTD_chemicals_diseases.tsv.gz": CTD + "CTD_chemicals_diseases.tsv.gz",
    "ontology/doid.obo": "https://raw.githubusercontent.com/DiseaseOntology/"
                         "HumanDiseaseOntology/main/src/ontology/doid.obo",
}

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(ROOT)} already present")
        return
    ...
    tmp = dest.with_suffix(dest.suffix + ".part")
    ...
    tmp.replace(dest)
```

Good practice here: downloads are **idempotent** (existing files are skipped), written to a `.part` file and renamed only when complete (no half-downloaded file can masquerade as complete), and sent with a descriptive `User-Agent`. The docstring states *why* each file is needed and that the chemical–disease file is "used only for case-study validation, never for training" — provenance and intent recorded at the point of download.

Gaps you should close in your own copy:
- **Two inputs are not in `FILES`.** `02_build_features.py` reads `data/raw/ontology/mondo.obo` and `data/raw/medgen/MedGenIDMappings.txt.gz`, but `01_download_data.py` does not fetch them. On a fresh machine the pipeline would fail at `parse_mondo()`. The sources are MONDO's permanent URL `http://purl.obolibrary.org/obo/mondo.obo` (which redirects to the latest GitHub release) and NCBI's FTP site (`https://ftp.ncbi.nlm.nih.gov/pub/medgen/MedGenIDMappings.txt.gz`); both resolved when this chapter was written.
- **No dates or checksums are recorded.** CTD files have no version number, so the download date *is* the version. The manifest of Section 4.10 fixes this. (The versions in the current files: CTD reports created 29 September 2026; MONDO `releases/2026-09-01`; DOID `releases/2026-09-30`.)
- Re-running after deleting `data/raw/` will fetch *newer* CTD/MONDO versions, so results can change. Keep the old files (or their checksums) with your results.

### 5.2 `load_benchmark` — the IDs we start from

```python
drugs = [str(x[0]) for x in m["Wrname"].ravel()]
# 'D102100' -> OMIM 102100 (the 'D' just marks it as a disease)
omims = [str(x[0])[1:] for x in m["Wdname"].ravel()]
A = m["didr"].T.astype(np.float32)          # -> drugs x diseases
```

Two small identifier facts: the `.mat` files prefix every OMIM number with `D` (a local convention of the benchmark, *not* a MeSH ID — another "leading letter" trap), and the association matrix is stored diseases × drugs, so it is transposed.

### 5.3 `pubchem_drugs` — DrugBank → PubChem

```python
txt = _get(f"{PUG}/substance/sourceid/DrugBank/{db}/cids/TXT")
cid = txt.split()[0] if txt else ""
name = ""
if not cid:   # biologics: no compound, but the substance still has a name
    syn = _get(f"{PUG}/substance/sourceid/DrugBank/{db}/synonyms/TXT")
    name = syn.splitlines()[0].strip() if syn else ""
...
time.sleep(0.21)                      # stay under PubChem's 5 req/s limit
...
txt = _get(f"{PUG}/compound/cid/{chunk}/property/SMILES,InChIKey,Title/CSV")
```

- It uses **strategy 1** (authoritative exact xref): DrugBank's own deposit.
- It respects the **rate limit** (0.21 s between calls ⇒ < 5 per second), **retries** on 503 with increasing waits (`_get`), treats 404 as "not found", **batches** property requests by 100 CIDs, and **caches** everything in `data/interim/pubchem_drugs.csv`, so re-runs send no requests for known drugs.
- `txt.split()[0]` takes the **first** CID when the substance standardizes to several (possible for mixtures). That is a silent many-to-one choice; it is harmless for our drugs but worth a comment or a count.
- Names: when the PubChem compound title looks odd (contains `,` or `;`, looks like `med.21724`, or is empty), the code replaces it with the first synonym of DrugBank's *own* substance, i.e. DrugBank's name. This heuristic missed `Npc209773` (atropine, `DB00572`), whose title has none of those patterns, and that drug consequently fails the name route to CTD. A simpler rule — *always* prefer the first DrugBank synonym — would avoid it at the cost of one extra request per drug.

Result: 667 of 682 drugs have a CID, SMILES and InChIKey.

### 5.4 `parse_mondo` and `disease_table` — OMIM → names, MONDO, MeSH, CTD

```python
for x in v["xrefs"]:
    if x.startswith("OMIM:"):
        omim2mondo[x[5:]].append(k)
    elif x.startswith("MESH:"):
        mondo2mesh[k].append(x)
```

`parse_mondo` keeps only live `MONDO:` terms and reads xrefs by **prefix**, which correctly excludes `OMIMPS:` (phenotypic series). It does not read the qualifier; in the current release every OMIM xref on a live term is `equivalentTo` or `equivalentObsolete`, so nothing wrong slips in, but a future release with `relatedTo` OMIM xrefs would be accepted silently.

```python
mg_omim = mg[mg.src == "OMIM"].drop_duplicates("src_id").set_index("src_id")
mg_mesh = mg[mg.src == "MeSH"].groupby("cui").src_id.apply(list).to_dict()
...
for i in [r.DiseaseID] + r.AltDiseaseIDs.split("|"):
    if i.startswith("OMIM:"):
        omim2ctd[i[5:]].append(r.DiseaseID)
...
ctd = set(omim2ctd.get(o, [])) | {m for m in mesh if m in medic_ids}
name = mg_omim.loc[o, "name"] if o in mg_omim.index else (
    terms[mondo[0]]["name"] if mondo else "")
```

- **Filtering on `src == "OMIM"`** is exactly the namespace discipline of Sections 4.4 and 4.8: it excludes `OMIM included` concepts and MedGen's colliding internal numbers. `drop_duplicates("src_id")` would hide a one-to-many problem, but for our 405 diseases the relation is 1:1, so nothing is dropped.
- MeSH IDs are collected from two routes (MONDO xrefs and MedGen's CUI → MeSH rows) and **unioned**; CTD disease IDs come from MEDIC (primary and alternative IDs) plus any of those MeSH IDs that exist in MEDIC. Union is the right choice here because the goal is *recall* of disease genes, and each route covers different diseases (the MedGen route alone gives MeSH IDs for only 291 of 405; Exercise 12).
- **Name precedence:** MedGen's equivalent concept first, MONDO's term name second. Result: 410 of 415 diseases named (405 from MedGen, 5 from MONDO — the `equivalentObsolete` ones); 404 placed in MONDO; 401 with at least one CTD disease ID; 50 diseases have more than one CTD ID (e.g. 104300 → `MESH:C536594|MESH:D000544`).

### 5.5 `ctd_chemical_map` — DrugBank → CTD chemical

```python
ch["ChemicalID"] = ch.ChemicalID.str.replace("MESH:", "", regex=False)
by_cid = {c: i for c, i in zip(ch.PubChemCID, ch.ChemicalID) if c}
by_ik = {k: i for k, i in zip(ch.InChIKey, ch.ChemicalID) if k}
by_name = {}
for r in ch.itertuples():
    for n in [r.ChemicalName] + r.MESHSynonyms.split("|"):
        if n:
            by_name.setdefault(n.lower(), r.ChemicalID)
out = {}
for db, r in drugs_pc.iterrows():
    cid = by_cid.get(r.cid) or by_ik.get(r.inchikey) or by_name.get(str(r["name"]).lower())
```

The design is a textbook cascade — exact ID, then structure key, then name — and it correctly strips `MESH:` so that the IDs match the interaction file, and it skips empty keys (`if c`, `if k`, `if n`). But with the CTD release in `data/raw/` (report of 29 September 2026):

1. `by_cid` is keyed by strings like `"CID:126941"`, while `r.cid` is `"126941"`: **the CID step never matches**.
2. `by_ik` is **empty**, because the `InChIKey` column is empty.
3. So all **643** mappings come from `by_name` (563 via a CTD primary name, 80 via a MeSH synonym; Exercise 11).

Is the resulting mapping bad? No — Section 4.7 shows that, where the two routes overlap, the name route was *right* in the conflicting cases. But the code's behaviour differs from its docstring ("via PubChem CID, then InChIKey, then name"), and you must report the method that actually produced the mapping. If you repair the CID step (strip `CID:`), do it together with a conflict check, or you will import errors such as methotrexate → "Folic Acid Antagonists". Two further details: `dict` comprehensions keep the **last** CTD record for a duplicated CID and `setdefault` keeps the **first** record for a duplicated name — both are silent tie-breaks (there are no ambiguous names among our drugs, but there are 8 multi-record CIDs).

### 5.6 `ctd_drug_genes` and `ctd_disease_genes` — gene profiles

```python
for chunk in pd.read_csv(RAW / "ctd" / "CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#",
                         names=cols, usecols=["ChemicalID", "GeneSymbol", "OrganismID"],
                         dtype=str, chunksize=500_000):
    chunk = chunk[(chunk.OrganismID == "9606") & chunk.ChemicalID.isin(wanted)]
```

Chunked reading keeps memory low; `OrganismID == "9606"` keeps human interactions only. Genes are joined by **symbol** within one CTD release, which is safe (one release uses one symbol set); comparing symbols across releases or sources would not be. Literature bias is visible in the numbers: methotrexate has 1,896 human interacting genes, aspirin 581, donepezil 19; 578 of 682 drugs have at least one.

```python
gd = gd[gd.DirectEvidence.str.contains("marker/mechanism")]
```

This keeps `marker/mechanism` rows **and** the 310 rows labelled `marker/mechanism|therapeutic`, and drops the 1,961 therapeutic-only rows. The 310 kept rows do have independent mechanistic evidence, so keeping them is defensible; just describe the rule precisely ("links with marker/mechanism evidence, regardless of additional therapeutic evidence"). Disease gene sets are collected both by CTD disease ID and by the `OmimIDs` column; 279 of 415 diseases end up with at least one gene.

### 5.7 `ctd_curated_chem_disease` — the validation table

```python
chunk = chunk[chunk.DirectEvidence.notna() & chunk.ChemicalID.isin(wanted)]
```

`DirectEvidence.notna()` keeps **curated** rows only (inferred rows have an empty field, read as NaN), for our 643 mapped chemicals: 38,859 rows covering 614 chemicals. The table is written to `data/interim/ctd_curated_chem_disease.csv` and used **only** by `06_case_study.py`. Exercise 13 measures why this table must never be an input: 19.7% of Cdataset's known links have a curated therapeutic CTD link, against 0.5% of random unknown pairs.

### 5.8 `scripts/06_case_study.py` — external evidence look-ups

```python
def ctd_evidence(db, omim):
    ids = set(filter(None, dis_tab.loc[omim, "ctd_ids"].split("|"))) if omim in dis_tab.index else set()
    rows = ctd[ctd.drugbank_id.str.contains(db) &
               (ctd.DiseaseID.isin(ids) | ctd.OmimIDs.str.split("|").apply(lambda x: omim in x))]
    return "|".join(sorted(set(rows.DirectEvidence))) or "-"
```

The disease side uses the same OMIM → CTD mapping as the features (both MEDIC/MeSH IDs and CTD's own `OmimIDs` column). Because 104300 maps to the *general* `MESH:D000544`, a "CTD-confirmed" prediction for Alzheimer disease type 1 really means "CTD links this drug to Alzheimer disease in general" — state that granularity in the paper.

```python
r = requests.get("https://clinicaltrials.gov/api/v2/studies",
                 params={"query.cond": disease, "query.intr": drug,
                         "countTotal": "true", "pageSize": 1}, timeout=30)
time.sleep(0.3)
return r.json().get("totalCount")
```

A correct minimal API v2 call: `pageSize=1` because only the count is needed, `countTotal=true` to get it, a timeout, a pause between calls, and `None` (not 0) on failure so that "lookup failed" is not confused with "no trials". `trial_condition()` simplifies OMIM-style names ("Alzheimer disease type 1" → "Alzheimer disease") because trial records use plain clinical names. The case-study tables should carry the query date.

### 5.9 The project's mapping coverage, verified

| Step | Count | Source of the number |
|---|---|---|
| Unique drugs (F ∪ C) | 682 | `data/interim/drugs_all.csv` |
| …with PubChem CID, SMILES, InChIKey | 667 (97.8%) | same |
| …mapped to a CTD chemical (all by name) | 643 (94.3%) | same, Section 4.7 |
| …with ≥ 1 human CTD gene | 578 (84.8%) | `n_genes` column |
| Unique diseases (F ∪ C) | 415 | `data/interim/diseases_all.csv` |
| …named | 410 (405 MedGen + 5 MONDO) | Section 4.4 |
| …placed in MONDO | 404 (97.3%) | `mondo` column |
| …with ≥ 1 CTD disease ID | 401 | `ctd_ids` column |
| …with ≥ 1 curated marker/mechanism gene | 279 (67.2%) | `n_genes` column |

(Per-dataset percentages in HOW_IT_WORKS section 2.3 are computed over each benchmark's own drugs and diseases, so they differ slightly from these pooled ones.)

---

## 6. Common mistakes and misconceptions

1. **"An exact ID match is always right."** Cross-references can point to classes, products or combinations (methotrexate's CID → "Folic Acid Antagonists"). Check target names and fan-in.
2. **"If the code didn't crash, the join worked."** Format mismatches (`CID:2244` vs `2244`) return zero matches without an error. Always print or assert the match count of every join step.
3. **Joining on bare numbers.** `602439` is both an OMIM number and an unrelated MedGen internal ID. Carry the namespace.
4. **Treating `OMIM included` (or `relatedTo`) as equivalence.** It is a broader/narrower or merely related concept — sometimes the opposite (the *protective* allele concept for 104300).
5. **Letting empty strings or NaN act as keys.** They match each other (the 15 false "matches" in Section 4.7), and `str(NaN)` becomes the string `"nan"`, which then matches other `"nan"`s.
6. **Using InChIKey block 1 as if it were the full key.** It merges enantiomers and geometric isomers — different drugs with different indications.
7. **Forgetting salts.** Metformin and metformin hydrochloride have different InChIKeys; normalize to the parent if parent-level identity is what you want, and say so.
8. **Joining on MeSH tree numbers** or on gene symbols across releases. Tree numbers move every year; symbols get renamed. Join on descriptor IDs and NCBI Gene IDs.
9. **Silent tie-breaking** (`dict` comprehensions, `setdefault`, `drop_duplicates`, `split()[0]`). Each hides a one-to-many relation. Count the cases first; then choose a rule deliberately.
10. **Using CTD inferred links as if they were evidence.** They are computed from other links; they add no independent support, and as features they can leak labels.
11. **Using CTD therapeutic chemical–disease links (or any indication table: ChEMBL indications, DrugBank indications, SIDER, …) as model inputs.** They are labels in disguise.
12. **Reading a ClinicalTrials.gov count as efficacy.** It counts registered studies matching search terms, of any design and outcome.
13. **Not recording the download date.** CTD has no version numbers; without the date (or the `Report created` stamp) your results cannot be tied to a data release.
14. **Assuming licence-free means attribution-free.** CTD requires citation and notification; MONDO (CC BY) requires attribution; OMIM content cannot be redistributed without a licence.
15. **Hammering an API.** More than 5 requests/second to PubChem gets you throttled and possibly blocked; batch and cache instead.

---

## 7. Exercises

Difficulty: ★ conceptual, ★★ practical (paper and small computations), ★★★ coding. Solutions with code were executed from the project root; outputs shown are real.

**Exercise 1 (★).** For each string, name the database/namespace and the kind of entity: (a) `DB00563` (b) `2244` in a PubChem context (c) `D001241` (d) `D00109` (e) `C536594` (f) `C1863052` (g) `CN123456` (h) `MONDO:0007088` (i) `HGNC:9604` (j) `P23219` (k) `NCT01234567` (l) `CHEMBL25` (m) `OMIMPS:104300`-style IDs.

<details><summary>Solution</summary>

(a) DrugBank primary drug ID (a drug). (b) PubChem Compound ID, CID (a standardized structure; aspirin). (c) MeSH descriptor *Aspirin* (a chemical heading; also CTD's chemical ID). (d) KEGG DRUG entry *Aspirin* — note it is "D" + **5** digits, while MeSH descriptors are "D" + 6 (or 9) digits; never guess the namespace from the letter alone. (e) MeSH Supplementary Concept Record (class 3, a disease: Alzheimer disease type 1). (f) UMLS/MedGen CUI (a concept: Alzheimer disease type 1). (g) MedGen-created CUI for a concept not in UMLS. (h) MONDO disease term. (i) HGNC gene ID (PTGS1). (j) UniProt protein accession (human COX-1). (k) ClinicalTrials.gov study (NCT number; this particular number is just a format example). (l) ChEMBL molecule ID (aspirin) — the same namespace also holds targets and assays. (m) OMIM phenotypic series, a grouping of OMIM phenotype entries, not an entry itself.

</details>

**Exercise 2 (★).** On omim.org you see `%600634`, `#104300`, `*104760` and `^144400` (hypothetical symbols for illustration). Which may be used as disease columns in a drug–disease matrix, and what would you do with each?

<details><summary>Solution</summary>

`%600634` — a confirmed Mendelian phenotype with unknown molecular basis: a valid disease column. `#104300` — a phenotype with known molecular basis: valid. `*104760` — a **gene** entry (APP): not a disease; if it appears among disease columns, it is a mapping error (perhaps the curator meant the associated phenotype `#104300`). `^144400` — removed or moved: follow OMIM's pointer to the entry it was moved to (if any) and re-map; if it was removed, drop it or keep it with an explicit "retired" flag, and report the number of such columns. (In our data, the MONDO qualifier `equivalentObsolete` is the machine-readable hint for this case.)

</details>

**Exercise 3 (★).** Explain in your own words why `substance/sourceid/DrugBank/DB00945/cids` can return more than one CID, and what the project does in that case. When would that choice be wrong?

<details><summary>Solution</summary>

A substance is the depositor's description. If DrugBank deposited a mixture, a salt or a multi-component product, PubChem's standardization creates a CID for the whole substance and may also return component CIDs (the `cids_type` option controls which). So the list can hold several CIDs. The project takes the first (`txt.split()[0]`). This is wrong if the first CID is a counter-ion or an excipient component rather than the active moiety, or if the "drug" is genuinely a mixture (conjugated estrogens) for which no single structure is right. Safer: request `cids_type=standardized`, count how many drugs return more than one CID, and inspect those by hand.

</details>

**Exercise 4 (★).** Two records have InChIKeys `SUBDBMMJDZJVOS-UHFFFAOYSA-N` and `SUBDBMMJDZJVOS-DEOSSOPVSA-N`. A third has `SUBDBMMJDZJVOS-DEOSSOPVSA-M`. What is the relationship between each pair?

<details><summary>Solution</summary>

All three share block 1, so they have the same skeleton (connectivity). The first has `UHFFFAOYSA` in block 2: no stereo layer — the racemate / stereo-undefined form (omeprazole). The second has a defined stereo layer (esomeprazole, the S-enantiomer). The second and third are identical in blocks 1 and 2 and differ only in block 3 (`N` neutral vs `M`, one proton removed): the same stereoisomer in a different protonation state, e.g. the anion of a salt such as esomeprazole magnesium after the counter-ion is removed. Whether to treat 2 and 3 as "the same drug" is a modelling decision; treating 1 and 2 as the same would merge two different medicines.

</details>

**Exercise 5 (★).** For each CTD-derived feature proposal, say whether it is acceptable as a model input for predicting drug–disease indications, and why: (a) Jaccard similarity of drugs' curated human gene sets; (b) disease gene sets including inferred gene–disease links; (c) a binary feature "CTD has a therapeutic link between this drug and this disease"; (d) disease gene sets from marker/mechanism links; (e) a drug–disease score counting inferred chemical–disease links.

<details><summary>Solution</summary>

(a) Acceptable: it describes drug biology (with the caveat that some interactions were studied *because* a drug treats a disease). (b) Not acceptable: inferred gene–disease links are made by chaining gene–chemical and chemical–disease curated links, so a drug's therapeutic link to a disease puts that drug's genes into the disease's gene set — label information. (c) Not acceptable: it is (nearly) the label; Exercise 13 shows a 40-fold enrichment among known links. (d) Acceptable: disease biology, the project's choice. (e) Not acceptable: inferred chemical–disease links are generated from curated chemical–gene and gene–disease links, and many of the curated chemical–disease links behind CTD's other inferences are therapeutic; also it duplicates what the model is supposed to infer, so any gain is not a contribution of the model.

</details>

**Exercise 6 (★★).** A left table has keys with multiplicities `{a: 1, b: 1, c: 1, d: 1}`. The right table has `{a: 3, b: 1, c: 0, d: 2}`. How many rows do an inner join and a left join produce? Which pandas `validate=` value would fail?

<details><summary>Solution</summary>

Inner: $1\cdot3 + 1\cdot1 + 1\cdot0 + 1\cdot2 = 6$. Left: $3 + 1 + 1 + 2 = 7$ (c kept once with missing values). Left keys are unique and right keys are not, so `validate="one_to_one"` and `validate="many_to_one"` fail; `"one_to_many"` passes (and documents that you *expect* fan-out).

</details>

**Exercise 7 (★★).** Write the PUG-REST URLs (after the prolog) for: (a) the InChIKeys of CIDs 2244 and 126941 in one request, as CSV; (b) the CIDs whose name is "metformin hydrochloride"; (c) the parent compound of a salt CID `X`; (d) the list of depositors of CID 2244; (e) the synonyms DrugBank deposited for `DB00563`.

<details><summary>Solution</summary>

(a) `/compound/cid/2244,126941/property/InChIKey/CSV` (b) `/compound/name/metformin%20hydrochloride/cids/TXT` (URL-encode the space) (c) `/compound/cid/X/cids/TXT?cids_type=parent` (d) `/compound/cid/2244/xrefs/SourceName/JSON` (e) `/substance/sourceid/DrugBank/DB00563/synonyms/TXT`.

</details>

**Exercise 8 (★★).** Design (on paper) an OMIM → CTD-disease mapping cascade that reports, for each disease, the route used and a confidence level. Use only the sources in `data/raw/`.

<details><summary>Solution</summary>

1. **MEDIC direct:** OMIM appears as a primary `DiseaseID` (`OMIM:…`) → confidence *exact*.
2. **MONDO equivalent → MeSH in MEDIC:** OMIM xref with qualifier `equivalentTo` on a live MONDO term; that term's `MESH:` xrefs that exist in MEDIC → *exact* if the MeSH xref is also `equivalentTo`, otherwise *related*.
3. **MedGen equivalent concept → MeSH:** rows with `src == "OMIM"` give the CUI; the CUI's `MeSH` rows → *exact* (same concept).
4. **MEDIC alternative ID:** OMIM listed in `AltDiseaseIDs` of a MeSH term → *broader* (typically a general disease covering the OMIM subtype, e.g. 104300 → D000544).
5. **Obsolete handling:** if the MONDO xref is `equivalentObsolete`, flag the OMIM entry as retired but keep the MONDO-based mapping.
6. Report per route: number of diseases mapped, number with more than one CTD ID, and the union used. Keep the confidence column so that analyses can be repeated with *exact only*.

</details>

**Exercise 9 (★★).** You run `06_case_study.py` today and get 208 donepezil × Alzheimer trials; a colleague gets 211 next month. Is something wrong? What should the paper say? How would you retrieve *all* matching NCT IDs rather than the count?

<details><summary>Solution</summary>

Nothing is wrong: the registry is updated every weekday, so counts drift. The paper should report the query (parameters) and the query date, e.g. "ClinicalTrials.gov API v2, `query.cond`/`query.intr`, accessed 1 October 2026", and ideally the `dataTimestamp` from `/api/v2/version`. To get all IDs: request `/api/v2/studies?query.cond=…&query.intr=…&fields=NCTId&pageSize=1000`, then repeat with `pageToken=<nextPageToken>` until no `nextPageToken` is returned, pausing between pages and caching the result.

</details>

**Exercise 10 (★★★).** Write `normalize_curie(value, prefix)` and a `checked_join(left, right, on, validate, min_match)` that drops empty keys, asserts the cardinality and the coverage, and prints the row counts. Use them to join our drugs to CTD chemicals by CID.

<details><summary>Solution</summary>

```python
import re
import pandas as pd

def normalize_curie(value, default_prefix):
    """'CID:2244' -> 'PUBCHEM.COMPOUND:2244', '2244' -> same; 'MESH:D001241' / 'D001241' -> 'MESH:D001241'."""
    if pd.isna(value) or str(value).strip() == "":
        return None                                     # empty / NaN is NOT a key
    v = str(value).strip()
    m = re.fullmatch(r"([A-Za-z.]+):(.+)", v)
    local = m.group(2) if m else v
    return f"{default_prefix}:{local}"

def checked_join(left, right, on, validate, min_match=0.5):
    """Left join that refuses empty keys, asserts cardinality and reports coverage."""
    r = right[right[on].notna()]                        # drop empty keys on the right
    out = left.merge(r, on=on, how="left", validate=validate, indicator=True)
    cov = (out.drop_duplicates(on)["_merge"] == "both").sum() / left[on].nunique()
    print(f"join on {on!r}: {len(left)} -> {len(out)} rows, coverage {cov:.1%}")
    assert cov >= min_match, f"coverage {cov:.1%} below {min_match:.0%} - format mismatch?"
    return out.drop(columns="_merge")

cols = ["ChemicalName", "ChemicalID", "CasRN", "PubChemCID", "PubChemSID", "DTXSID", "InChIKey",
        "Definition", "ParentIDs", "TreeNumbers", "ParentTreeNumbers", "MESHSynonyms",
        "CTDCuratedSynonyms"]
ch = pd.read_csv("data/raw/ctd/CTD_chemicals.tsv.gz", sep="\t", comment="#", names=cols,
                 dtype=str, usecols=["ChemicalID", "ChemicalName", "PubChemCID"])
drugs = pd.read_csv("data/interim/drugs_all.csv", dtype=str)
drugs["cid_key"] = drugs.cid.map(lambda x: normalize_curie(x, "PUBCHEM.COMPOUND"))
ch["cid_key"] = ch.PubChemCID.map(lambda x: normalize_curie(x, "PUBCHEM.COMPOUND"))
print(drugs.cid_key.head(2).tolist(), ch.cid_key.dropna().head(2).tolist())
try:
    checked_join(drugs[["drugbank_id", "cid_key"]], ch[["cid_key", "ChemicalID"]],
                 on="cid_key", validate="many_to_one", min_match=0.2)
except pd.errors.MergeError as e:
    print("MergeError:", str(e).splitlines()[0])
out = checked_join(drugs[["drugbank_id", "cid_key"]], ch[["cid_key", "ChemicalID"]],
                   on="cid_key", validate="many_to_many", min_match=0.2)
```

Output:

```text
['PUBCHEM.COMPOUND:657181', nan] ['PUBCHEM.COMPOUND:2836600', 'PUBCHEM.COMPOUND:656732']
MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
join on 'cid_key': 682 -> 690 rows, coverage 34.2%
```

The `many_to_one` assertion fails — correctly, because 8 CIDs have two CTD records — so we must decide how to handle fan-out rather than discovering it later. With `many_to_many` the join runs: 682 rows become 690 (8 duplicated drugs) and 228 of the 667 distinct CIDs (34.2%) match. An earlier draft of this solution used `value is None` instead of `pd.isna(value)` and produced the key `"PUBCHEM.COMPOUND:nan"` for every missing CID: the empty-key pitfall in another disguise.

</details>

**Exercise 11 (★★★).** How many of our drugs match a CTD chemical by its *primary* name, how many only through a MeSH synonym, and are any drug names ambiguous (matching more than one CTD record)?

<details><summary>Solution</summary>

```python
from collections import defaultdict
import pandas as pd

cols = ["ChemicalName", "ChemicalID", "CasRN", "PubChemCID", "PubChemSID", "DTXSID", "InChIKey",
        "Definition", "ParentIDs", "TreeNumbers", "ParentTreeNumbers", "MESHSynonyms",
        "CTDCuratedSynonyms"]
ch = pd.read_csv("data/raw/ctd/CTD_chemicals.tsv.gz", sep="\t", comment="#", names=cols,
                 dtype=str, usecols=["ChemicalName", "ChemicalID", "MESHSynonyms"]).fillna("")
primary, synonym = defaultdict(set), defaultdict(set)
for r in ch.itertuples():
    primary[r.ChemicalName.lower()].add(r.ChemicalID)
    for s in filter(None, r.MESHSynonyms.split("|")):
        synonym[s.lower()].add(r.ChemicalID)

drugs = pd.read_csv("data/interim/drugs_all.csv", dtype=str).fillna("")
kinds = {"primary name": 0, "synonym only": 0, "ambiguous": 0, "no match": 0}
examples = []
for name in drugs.name.str.lower():
    cands = primary.get(name, set()) | synonym.get(name, set())
    if len(cands) > 1:
        kinds["ambiguous"] += 1
        examples.append((name, sorted(cands)))
    elif name in primary:
        kinds["primary name"] += 1
    elif name in synonym:
        kinds["synonym only"] += 1
    else:
        kinds["no match"] += 1
print(kinds)
print("ambiguous examples:", examples[:3])
```

Output:

```text
{'primary name': 563, 'synonym only': 80, 'ambiguous': 0, 'no match': 39}
ambiguous examples: []
```

563 + 80 = 643, the project's mapping count, and no drug name is ambiguous, so the `setdefault` tie-break in `ctd_chemical_map` never fires for our drugs. The 39 unmatched drugs are what an InChIKey or repaired CID route could try to recover.

</details>

**Exercise 12 (★★★).** Reproduce the MedGen OMIM → CUI → MeSH route used inside `disease_table` and measure its coverage and cardinality for our diseases.

<details><summary>Solution</summary>

```python
import pandas as pd

mg = pd.read_csv("data/raw/medgen/MedGenIDMappings.txt.gz", sep="|", dtype=str, usecols=[0, 1, 2, 3])
mg.columns = ["cui", "name", "src_id", "src"]
ours = pd.read_csv("data/interim/diseases_all.csv", dtype=str).omim

omim2cui = mg[(mg.src == "OMIM") & mg.src_id.isin(ours)][["src_id", "cui"]]
cui2mesh = mg[mg.src == "MeSH"][["cui", "src_id"]].rename(columns={"src_id": "mesh"})
m = omim2cui.merge(cui2mesh, on="cui", how="left", validate="one_to_many")
n_mesh = m.groupby("src_id").mesh.nunique()
print(f"OMIM IDs with a CUI: {len(omim2cui)}; with >=1 MeSH ID via the CUI: {(n_mesh > 0).sum()}")
print("MeSH IDs per OMIM ID:", n_mesh.value_counts().sort_index().to_dict())
print(m[m.src_id == "104300"].to_string(index=False))
```

Output:

```text
OMIM IDs with a CUI: 405; with >=1 MeSH ID via the CUI: 291
MeSH IDs per OMIM ID: {0: 114, 1: 291}
src_id      cui    mesh
104300 C1863052 C536594
```

Only 291 of 405 diseases get a MeSH ID through MedGen, all with exactly one; for 104300 the route gives the specific SCR `C536594`, not the general descriptor. That is why the project unions this route with MONDO's MeSH xrefs and MEDIC's OMIM IDs.

</details>

**Exercise 13 (★★★).** Quantify how much label information a CTD curated *therapeutic chemical–disease* link carries: what share of Cdataset's known links have one, versus random unknown pairs?

<details><summary>Solution</summary>

```python
import numpy as np
import pandas as pd

z = np.load("data/processed/Cdataset.npz")
A, drugs, omims = z["A"], z["drug_ids"], z["disease_ids"]
ctd = pd.read_csv("data/interim/ctd_curated_chem_disease.csv", dtype=str).fillna("")
dis = pd.read_csv("data/interim/diseases_all.csv", dtype=str).fillna("").set_index("omim")

# (drugbank_id, CTD disease ID) pairs with curated THERAPEUTIC evidence
ther = ctd[ctd.DirectEvidence == "therapeutic"]
pairs = {(db, d) for dbs, d in zip(ther.drugbank_id, ther.DiseaseID) for db in dbs.split("|") if db}

def has_ther(db, omim):
    ids = set(filter(None, dis.loc[omim, "ctd_ids"].split("|"))) if omim in dis.index else set()
    return any((db, d) in pairs for d in ids)

known = [(drugs[i], omims[j]) for i, j in zip(*np.nonzero(A))]
hit = np.mean([has_ther(db, o) for db, o in known])
rng = np.random.default_rng(0)
zeros = np.argwhere(A == 0)[rng.choice(int((A == 0).sum()), 5000, replace=False)]
base = np.mean([has_ther(drugs[i], omims[j]) for i, j in zeros])
print(f"known Cdataset links with a CTD curated therapeutic link: {hit:.1%} of {len(known)}")
print(f"random unknown pairs with one:                            {base:.1%} of 5000")
```

Output:

```text
known Cdataset links with a CTD curated therapeutic link: 19.7% of 2532
random unknown pairs with one:                            0.5% of 5000
```

A known link is about 40 times more likely than a random unknown pair to have a curated therapeutic CTD link. As a feature it would be a near-copy of the label for a fifth of the positives — classic illegitimate-feature leakage (Unit E1). As *validation* for new predictions it is useful precisely because it is label-like, which is how `06_case_study.py` uses it.

</details>

**Exercise 14 (★★★).** Now test gene–disease evidence. For Fdataset, compare how often a drug's CTD genes overlap a disease's *therapeutic-only* genes versus its *marker/mechanism* genes, for known links and for random unknown pairs.

<details><summary>Solution</summary>

```python
from collections import defaultdict
import numpy as np
import pandas as pd

z = np.load("data/processed/Fdataset.npz")
A, drugs, omims = z["A"], z["drug_ids"], z["disease_ids"]
drug_tab = pd.read_csv("data/interim/drugs_all.csv", dtype=str).fillna("").set_index("drugbank_id")
dis = pd.read_csv("data/interim/diseases_all.csv", dtype=str).fillna("").set_index("omim")

# drug -> human genes (curated chemical-gene interactions), as in ctd_drug_genes()
chem = dict(zip(drug_tab.index, drug_tab.ctd_chemical))
wanted, cgenes = set(chem.values()) - {""}, defaultdict(set)
cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
        "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
for ch in pd.read_csv("data/raw/ctd/CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#", names=cols,
                      usecols=["ChemicalID", "GeneSymbol", "OrganismID"], dtype=str,
                      chunksize=500_000):
    ch = ch[(ch.OrganismID == "9606") & ch.ChemicalID.isin(wanted)]
    for c, g in zip(ch.ChemicalID, ch.GeneSymbol):
        cgenes[c].add(g)
drug_genes = {d: cgenes.get(chem.get(d, ""), set()) for d in drugs}

# disease -> genes, split by evidence type
gd = pd.read_csv("data/raw/ctd/CTD_curated_genes_diseases.tsv.gz", sep="\t", comment="#",
                 names=["GeneSymbol", "GeneID", "DiseaseName", "DiseaseID", "DirectEvidence",
                        "OmimIDs", "PubMedIDs"], dtype=str).fillna("")
def disease_genes(mask):
    by_ctd = defaultdict(set)
    for r in gd[mask].itertuples():
        by_ctd[r.DiseaseID].add(r.GeneSymbol)
    return {o: set().union(*[by_ctd.get(c, set()) for c in dis.loc[o, "ctd_ids"].split("|") if c])
            for o in omims}
only_ther = disease_genes(gd.DirectEvidence == "therapeutic")
marker = disease_genes(gd.DirectEvidence.str.contains("marker/mechanism"))

rng = np.random.default_rng(0)
pos = list(zip(*np.nonzero(A)))
neg = [tuple(x) for x in np.argwhere(A == 0)[rng.choice(int((A == 0).sum()), 20000, replace=False)]]
for label, dg in [("therapeutic-only gene-disease", only_ther), ("marker/mechanism gene-disease", marker)]:
    share = lambda P: np.mean([len(drug_genes[drugs[i]] & dg[omims[j]]) > 0 for i, j in P])
    p, n = share(pos), share(neg)
    print(f"{label:30s}: drug genes overlap disease genes for {p:5.1%} of known links, "
          f"{n:5.1%} of unknown pairs  (lift {p / n:4.1f}x)")
```

Output:

```text
therapeutic-only gene-disease : drug genes overlap disease genes for  3.7% of known links,  1.3% of unknown pairs  (lift  2.8x)
marker/mechanism gene-disease : drug genes overlap disease genes for 21.6% of known links, 10.2% of unknown pairs  (lift  2.1x)
```

Both kinds of gene evidence are enriched among known links — that is the legitimate biological signal ("drugs treat diseases whose genes they touch") the `gene_d` view and the gene bridge rely on. The therapeutic-only links show a higher lift (2.8× vs 2.1×) while covering few diseases; their *definition* ("the gene is a therapeutic target in the disease") is derived from treatment knowledge, so the extra lift cannot be trusted as biology. The much larger danger is the *inferred* gene–disease file (3.2 GB, not downloaded), whose links are constructed through chemical–disease links. Excluding both keeps the inputs on the safe side of the line; the price is a small loss of possibly-legitimate signal.

</details>

**Exercise 15 (★★).** You want to publish a GitHub repository with the project's code and processed `.npz` files. For each item, may you include it? (a) DrugBank IDs; (b) SMILES fetched from PubChem; (c) disease names from MedGen; (d) a copy of `CTD_chem_gene_ixns.tsv.gz`; (e) the CTD-derived gene Jaccard matrices; (f) OMIM's `genemap2.txt`; (g) MONDO-derived semantic similarity.

<details><summary>Solution</summary>

(a) Yes: identifiers are facts, and the DrugBank vocabulary is CC0. (b) Yes: PubChem data are free to use; cite PubChem. (c) Yes, with citation of MedGen. (d) Better not: link to CTD's download page instead; CTD's terms require citation, linking and notification, and re-hosting their file adds nothing. (e) Yes, as derived data, with citation of CTD and notification per its terms. (f) No: OMIM bulk files cannot be redistributed without a licence from Johns Hopkins. (g) Yes, with attribution (MONDO is CC BY 4.0). In all cases put a `DATA_LICENSES.md` and the manifest (Section 4.10) in the repository.

</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (Unit D4)

### Q1. Why is InChIKey a good key to join chemical databases?

Because it is a **structure-derived, canonical, fixed-length identifier that no single database owns**.

1. **Canonical.** It is a hash of the *standard* InChI, which is defined so that every correct representation of the same molecule — any atom order, any drawing, any file format, any software — yields the same string. Two databases that both hold aspirin will compute the same key `BSYNRYMUTXBXSQ-UHFFFAOYSA-N` independently, without ever having exchanged identifiers. Database-specific IDs (CIDs, DrugBank IDs, ChEMBL IDs, MeSH IDs) can only be joined through someone's cross-reference table, and Section 4.7 shows that such tables can be wrong (CTD's methotrexate CID → "Folic Acid Antagonists") or formatted inconsistently (`CID:` prefixes).
2. **Neutral and open.** InChI is an IUPAC standard with free software; every major chemistry database publishes it, which is why UniChem uses it as the hub between dozens of sources.
3. **Fixed-length and index-friendly.** 27 ASCII characters, no special characters, ideal for hashing, database indices and even web search — unlike SMILES (not unique across toolkits), or full InChI strings (long and variable).
4. **Layered, so you can choose the level of identity.** Block 1 (skeleton) vs block 2 (stereo, isotopes) vs block 3 (protonation) let you decide whether enantiomers or protonation states count as "the same", explicitly rather than by accident. Section 4.6 found 12 skeleton-sharing groups among our drugs (e.g. omeprazole/esomeprazole); full-key matching keeps them apart, block-1 matching merges them.

Its **limits**, which a good answer also states: salts and mixtures are different molecules to InChI, so normalize to the parent first if needed; standard InChI handles only some tautomerism, so tautomers can get different keys; it says nothing about biologics without a defined small-molecule structure (none of our 9 biologics/mixtures has one); it inherits any error in the deposited structure; and a hash can in principle collide (negligible in practice). And it only works if *both* sides publish it — the current CTD file's InChIKey column is empty, so in this project InChIKey could not be used for the CTD join at all.

### Q2. Why did we exclude CTD *inferred* and *therapeutic* gene–disease links from the model's inputs?

Because both are partly **derived from knowledge of which drugs treat which diseases** — the very relation the model is asked to predict and on which it is evaluated. Using them would let information about the labels, including the hidden test links, enter the model through the features (data leakage; Kapoor & Narayanan's "illegitimate features", Unit E1).

- **Inferred gene–disease links** are computed by CTD by chaining: gene G interacts with chemical C (curated), and C is curated as associated with disease D ⇒ G is "inferred" to be associated with D via C. Many curated chemical–disease links are `therapeutic` — drug indications. So if drug C treats disease D (perhaps a link hidden in our test fold), all of C's genes are added to D's gene set. Then the `gene_d` disease similarity and, above all, the drug–gene–disease bridge (cosine of drug genes vs disease genes) would light up for exactly the pair (C, D) — the model would "predict" a test link because the answer was smuggled into the features. The evaluation would look excellent and mean nothing.
- **Therapeutic gene–disease links** mean, by CTD's definition, that the gene *is or may be a therapeutic target in the treatment of the disease*. Curators record them largely from studies where a drug acting on that gene treats the disease. They therefore encode drug–disease treatment knowledge one step removed. Exercise 14 shows the effect on Fdataset: a drug's genes overlap a disease's therapeutic-only genes 2.8 times more often for known links than for random pairs, more than for mechanism genes (2.1×), even though the therapeutic set is small.
- By contrast, **curated `marker/mechanism` links** describe disease biology (biomarkers, causal genes) established independently of any particular treatment, so they are a legitimate description of the disease.

Two supporting points complete the answer. First, the same logic keeps **curated chemical–disease links** out of the inputs entirely: 19.7% of Cdataset's known links have a curated therapeutic CTD link versus 0.5% of random pairs (Exercise 13), so they are used only to *check* case-study predictions. Second, the exclusion is a choice with a cost (a little potentially-real signal is lost) and a residual risk (the 310 links with both evidence types are kept; chemical–gene interactions themselves come from a literature that studies drugs in their indications). The honest methods section states the rule exactly and mentions the residual risk.

---

## 9. Summary and cheat sheet

**The big idea.** Entity resolution is the foundation of every multi-source biomedical model. It fails silently, so build mappings that *measure themselves*: coverage, cardinality, conflicts, provenance.

**Identifier cheat sheet**

| Looks like | Is | Database |
|---|---|---|
| `DB00945` | drug | DrugBank (salts: `DBSALT…`) |
| `2244` (CID) / SID / AID | compound / deposited substance / assay | PubChem |
| `BSYNRYMUTXBXSQ-UHFFFAOYSA-N` | structure key (skeleton-stereo/version-protonation) | InChI (any database) |
| `CHEMBL25` | molecule (or target/assay!) | ChEMBL |
| `D00109` / `hsa:5742` | drug / gene | KEGG |
| `D001241` / `C536594` | MeSH descriptor / supplementary concept | MeSH (CTD chemicals and diseases) |
| `C10.228.140…` | position in the MeSH tree (not an ID) | MeSH |
| `104300` with `*`/`#`/`%`/`+`/`^` | gene / phenotype (basis known) / phenotype (unknown) / gene+phenotype / retired | OMIM |
| `C1863052` / `CN…` | concept | UMLS / MedGen |
| `MONDO:0007088`, `DOID:…` | disease term | MONDO, Disease Ontology |
| `5742`, `PTGS1`/`HGNC:9604`, `P23219` | gene, symbol/HGNC ID, protein | NCBI Gene, HGNC, UniProt |
| `NCT…` | clinical study | ClinicalTrials.gov |

**CTD in five lines.** Curated = stated in a paper; inferred = chained through a shared gene or chemical. `DirectEvidence` ∈ {`marker/mechanism`, `therapeutic`} (gene–disease can have both). Model inputs here: human chemical–gene interactions + marker/mechanism gene–disease links. Validation only: curated chemical–disease links. Never: inferred links, therapeutic gene–disease links.

**API etiquette.** PubChem: ≤ 5 requests/s, batch by 100, cache, retry on 503 with back-off. ClinicalTrials.gov v2: `query.cond`, `query.intr`, `countTotal=true`, `pageSize`/`pageToken`; record the query date.

**Resolution cascade.** Authoritative xref → curated xref (check qualifier, fan-in) → full InChIKey (parent-normalized) → hub service → normalized names + synonyms (check ambiguity) → manual review. Record the method per pair; compare routes; send conflicts to review.

**Join hygiene.** Normalize IDs to CURIEs; drop empty keys; filter on namespace; `merge(validate=…)`; print rows-in/rows-out and coverage for every join.

**Provenance.** Immutable `data/raw/`; manifest with URL, date, size, SHA-256, internal version, licence; cache API answers with dates; cite every source and its version.

---

## 10. Curated further resources

All links were checked on 1 October 2026. "Free" = free to read; "registration" = free account needed.

**Official documentation**
- [DrugBank release page](https://go.drugbank.com/releases/latest) — versions, file list, CC BY-NC 4.0 and CC0 Open Data terms. *Free (downloads need registration).*
- [PubChem PUG-REST tutorial](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial) — the best introduction to the URL grammar, with many examples. *Free.*
- [PubChem PUG-REST specification](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) — every input, operation and option (`cids_type`, `xrefs`, properties). *Free.*
- [PubChem programmatic access and usage policy](https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access) and [dynamic request throttling](https://pubchem.ncbi.nlm.nih.gov/docs/dynamic-request-throttling) — the 5 requests/s rule and the throttling header. *Free.*
- [PubChem Substance vs Compound documentation](https://pubchem.ncbi.nlm.nih.gov/docs/substances) — what depositors and standardization do. *Free.*
- [ChEMBL interface documentation](https://chembl.gitbook.io/chembl-interface-documentation) — IDs, data model, web services. *Free.*
- [UniChem](https://www.ebi.ac.uk/unichem/) — InChIKey-based cross-referencing of chemical IDs across sources. *Free.*
- [OMIM FAQ](https://www.omim.org/help/faq) — MIM numbering and entry symbols (`*`, `#`, `%`, `+`, `^`). *Free.*
- [OMIM downloads and licensing](https://www.omim.org/downloads) — which files need registration or a licence. *Free to read.*
- [MeSH record types](https://www.nlm.nih.gov/mesh/intro_record_types.html) and [MeSH tree structures](https://www.nlm.nih.gov/mesh/intro_trees.html) — descriptors, SCR classes, why tree numbers are not IDs. *Free.*
- [MeSH Browser](https://meshb.nlm.nih.gov/) — look up any `D…`/`C…` ID. *Free.*
- [CTD downloads](https://ctdbase.org/downloads/) and [CTD terms of use](https://ctdbase.org/about/legal.jsp) — file formats and conditions (opens in a browser; CTD blocks scripted access to its web pages). *Free.*
- [MedGen overview](https://www.ncbi.nlm.nih.gov/medgen/docs/overview/) — CUIs, `CN` identifiers, sources; [MedGen FTP](https://ftp.ncbi.nlm.nih.gov/pub/medgen/) for `MedGenIDMappings.txt.gz`. *Free.*
- [UMLS](https://www.nlm.nih.gov/research/umls/index.html) — the Metathesaurus and CUIs. *Free licence required.*
- [Mondo Disease Ontology](https://mondo.monarchinitiative.org/) and its [GitHub repository](https://github.com/monarch-initiative/mondo) — releases, xref qualifiers. *Free (CC BY 4.0).*
- [Human Disease Ontology](https://disease-ontology.org/) — DOID terms and OMIM xrefs. *Free (CC0).*
- [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api) and [search areas](https://clinicaltrials.gov/data-api/about-api/search-areas) — API v2 specification and which fields each query parameter searches. *Free.*
- [KEGG](https://www.genome.jp/kegg/), [UniProt](https://www.uniprot.org/), [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene), [HGNC](https://www.genenames.org/) — gene/protein/pathway identifiers. *Free (KEGG FTP paid).*
- [CLUE / Connectivity Map](https://clue.io/) — L1000 data and tools. *Registration.*
- [Bioregistry](https://bioregistry.io/) — the registry of identifier prefixes, their patterns and resolvers; the answer to "what is the correct CURIE prefix for X?". *Free.*

**Papers (reference descriptions to cite)**
- Knox C. et al. (2024) *DrugBank 6.0: the DrugBank Knowledgebase for 2024.* Nucleic Acids Res 52:D1265–D1275. [doi:10.1093/nar/gkad976](https://doi.org/10.1093/nar/gkad976) — *Free.*
- Kim S. et al. (2016) *PubChem Substance and Compound databases.* Nucleic Acids Res 44:D1202. [doi:10.1093/nar/gkv951](https://doi.org/10.1093/nar/gkv951) — the clearest explanation of SID vs CID. *Free.*
- Kim S. et al. (2025) *PubChem 2025 update.* Nucleic Acids Res 53:D1516–D1525. [doi:10.1093/nar/gkae1059](https://doi.org/10.1093/nar/gkae1059) — *Free.*
- Heller S.R. et al. (2015) *InChI, the IUPAC International Chemical Identifier.* J Cheminform 7:23. [doi:10.1186/s13321-015-0068-4](https://doi.org/10.1186/s13321-015-0068-4) — layers, InChIKey, limitations. *Free.*
- Chambers J. et al. (2013) *UniChem: a unified chemical structure cross-referencing and identifier tracking system.* J Cheminform 5:3. [doi:10.1186/1758-2946-5-3](https://doi.org/10.1186/1758-2946-5-3) — *Free.*
- Zdrazil B. et al. (2024) *The ChEMBL Database in 2023.* Nucleic Acids Res 52:D1180–D1192. [doi:10.1093/nar/gkad1004](https://doi.org/10.1093/nar/gkad1004) — *Free.*
- Amberger J.S. et al. (2019) *OMIM.org: leveraging knowledge across phenotype–gene relationships.* Nucleic Acids Res 47:D1038. [doi:10.1093/nar/gky1151](https://doi.org/10.1093/nar/gky1151) — *Free.*
- Davis A.P. et al. (2023) *Comparative Toxicogenomics Database (CTD): update 2023.* Nucleic Acids Res 51:D1257. [doi:10.1093/nar/gkac833](https://doi.org/10.1093/nar/gkac833) — the reference named in PREREQUISITES.md. *Free.*
- Davis A.P. et al. (2025) *Comparative Toxicogenomics Database's 20th anniversary: update 2025.* Nucleic Acids Res 53:D1328–D1334. [doi:10.1093/nar/gkae883](https://doi.org/10.1093/nar/gkae883) — current content statistics. *Free.*
- Davis A.P. et al. (2012) *MEDIC: a practical disease vocabulary used at the Comparative Toxicogenomics Database.* Database bar065. [doi:10.1093/database/bar065](https://doi.org/10.1093/database/bar065) — *Free.*
- King B.L. et al. (2012) *Ranking transitive chemical–disease inferences using local network topology in the Comparative Toxicogenomics Database.* PLoS ONE 7:e46524. [doi:10.1371/journal.pone.0046524](https://doi.org/10.1371/journal.pone.0046524) — how CTD's inference scores work. *Free.*
- Bodenreider O. (2004) *The Unified Medical Language System (UMLS).* Nucleic Acids Res 32:D267. [doi:10.1093/nar/gkh061](https://doi.org/10.1093/nar/gkh061) — *Free.*
- Vasilevsky N.A. et al. (2022) *Mondo: Unifying diseases for the world, by the world.* medRxiv. [doi:10.1101/2022.04.13.22273750](https://doi.org/10.1101/2022.04.13.22273750) — *Free.*
- Lamb J. et al. (2006) *The Connectivity Map.* Science 313:1929. [doi:10.1126/science.1132939](https://doi.org/10.1126/science.1132939) — *Paywalled (often free via institution).*
- Subramanian A. et al. (2017) *A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles.* Cell 171:1437. [doi:10.1016/j.cell.2017.10.049](https://doi.org/10.1016/j.cell.2017.10.049) — *Free.*
- Himmelstein D.S. et al. (2017) *Systematic integration of biomedical knowledge prioritizes drugs for repurposing* (Hetionet). eLife 6:e26726. [doi:10.7554/eLife.26726](https://doi.org/10.7554/eLife.26726) — a model of careful multi-database integration, provenance and licence tracking for repurposing. *Free.*

---

## 11. Glossary

- **AID** — PubChem BioAssay identifier.
- **Alternative (secondary) ID** — an older or merged identifier that still resolves to a current record.
- **Cardinality (of a mapping)** — how many targets each source entity maps to and vice versa: 1:1, 1:n, n:1, n:m.
- **Cascade (mapping)** — trying resolution methods in decreasing order of reliability, recording which one succeeded.
- **CID** — PubChem Compound identifier: one standardized unique structure.
- **ClinicalTrials.gov API v2** — the REST API of the trial registry (`/api/v2/studies`).
- **Coverage** — the share of your entities that obtained a mapping.
- **CTD** — Comparative Toxicogenomics Database: curated chemical–gene, chemical–disease and gene–disease relationships plus inferences.
- **CUI** — Concept Unique Identifier in UMLS/MedGen (`C` + 7 digits; `CN…` for MedGen-only concepts).
- **Curated link** — a relationship recorded by a curator from a publication (CTD "direct evidence").
- **CURIE** — compact URI, `prefix:local_id`.
- **Depositor (source)** — an organization that submits substances or assays to PubChem; each keeps its own *source ID*.
- **DirectEvidence** — CTD field for curated links: `marker/mechanism` or `therapeutic`.
- **DrugBank** — curated drug knowledge base; IDs `DB#####`; CC BY-NC 4.0 (Open Data subset CC0).
- **Entity resolution** — deciding which records in different sources denote the same real-world entity.
- **`equivalentObsolete`** — MONDO xref qualifier: equivalent concept, but the target ID has been obsoleted.
- **Granularity mismatch** — mapping between concepts at different levels of specificity (subtype ↔ general disease).
- **HGNC** — HUGO Gene Nomenclature Committee: official human gene symbols and `HGNC:` IDs.
- **Inferred link** — a CTD relationship computed by chaining two curated links through a shared gene or chemical.
- **InChI / InChIKey** — IUPAC canonical structure identifier / its 27-character hashed form (skeleton–stereo/flags–protonation).
- **Join explosion** — row multiplication when join keys are not unique: $\sum_k a_k b_k$ rows.
- **Manifest** — a file recording URL, date, size, checksum, internal version and licence of every raw input.
- **MedGen** — NCBI portal integrating disease concepts from UMLS, OMIM, HPO, MONDO and others.
- **MEDIC** — CTD's merged disease vocabulary (MeSH disease tree + OMIM).
- **MeSH descriptor / SCR / tree number** — main heading (`D…`) / supplementary concept record (`C…`) / position in the hierarchy (not an ID).
- **MIM number** — six-digit OMIM entry number; prefixes `*` gene, `#` phenotype with known basis, `%` phenotype with unknown basis, `+` gene and phenotype, `^` removed/moved.
- **MONDO** — Mondo Disease Ontology, a merged disease ontology and cross-reference hub (CC BY 4.0).
- **Namespace collision** — the same bare string meaning different things in different ID systems.
- **NCT number** — ClinicalTrials.gov study identifier.
- **OMIM** — Online Mendelian Inheritance in Man, catalogue of human genes and genetic phenotypes (© Johns Hopkins).
- **OMIMPS** — OMIM phenotypic series identifier, a separate namespace from OMIM entries.
- **Parent compound** — the main neutral organic component of a salt or mixture.
- **Provenance** — the documented origin and processing history of data.
- **PUG-REST** — PubChem's URL-based web service: input / operation / output.
- **Qualifier (mapping)** — the stated strength of a cross-reference (`equivalentTo`, `relatedTo`, …).
- **Rate limit / throttling** — the maximum request rate a service accepts; excess requests get HTTP 503 or blocking.
- **Retired / obsolete ID** — an identifier whose record was withdrawn, merged or moved.
- **SID** — PubChem Substance identifier: one depositor's record.
- **Source ID** — a depositor's own identifier for a PubChem substance (e.g. DrugBank's `DB00945`).
- **Standardization (PubChem)** — PubChem's processing of deposited substances into unique compounds.
- **UMLS** — Unified Medical Language System (NLM), merging >100 vocabularies into concepts.
- **UniChem** — EMBL-EBI service mapping chemical IDs across sources via InChIKey.
- **UniProt accession** — identifier of a protein record (e.g. `P23219`).
- **Version drift** — changes in a source's content or format between downloads that alter results.
- **Xref** — cross-reference from a record in one database to a record in another.
