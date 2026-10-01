# Learning track: everything you need for this project

This folder is a complete, self-contained course for the drug-repositioning
project. There is one in-depth chapter per topic, in the order you should study
them. Each chapter is written to be studied on its own, without needing outside
material. Every chapter ends with curated links if you want a second
explanation, but you should not *need* them.

## How each chapter is built

Every chapter follows the same structure, like a university course module:

1. **Learning objectives**: what you will be able to do afterwards
2. **Motivation**: why it matters *for this project*, with a real example
3. **Core theory**: intuition first, then definitions, maths and derivations, with worked examples by hand
4. **Code**: runnable Python examples (all tested) with the expected output
5. **In this project**: a walk through the exact project code that uses the idea
6. **Common mistakes**
7. **Exercises with full solutions** (click "Solution" to expand)
8. **Answers to the self-check questions** from `docs/PREREQUISITES.md`
9. **Cheat sheet**
10. **Further resources**: the best courses, books and papers, checked to work
11. **Glossary**

## Study order

Follow the numbers. Each chapter only assumes the ones before it.

| # | Chapter | Track | Est. time |
|---|---|---|---|
| 01 | [Python, NumPy & pandas for data work](01_A1_python_numpy_pandas.md) | Foundations | 8–10 h |
| 02 | [Linear algebra for ML and graphs](02_A2_linear_algebra.md) | Foundations | 12–15 h |
| 03 | [Probability & statistics](03_A3_probability_statistics.md) | Foundations | 10–12 h |
| 04 | [Supervised learning basics](04_B1_supervised_learning.md) | Machine learning | 10–12 h |
| 05 | [Evaluation & imbalanced data](05_B2_evaluation_imbalanced_data.md) | Machine learning | 8–10 h |
| 06 | [Pharmacology & drug repositioning](06_D1_pharmacology_drug_repositioning.md) | Biology / chemistry | 6–8 h |
| 07 | [Neural networks & PyTorch](07_B3_neural_networks_pytorch.md) | Machine learning | 12–15 h |
| 08 | [Graph theory basics](08_C1_graph_theory_basics.md) | Graphs | 6–8 h |
| 09 | [Random walks & propagation](09_C2_random_walks_propagation.md) | Graphs | 8–10 h |
| 10 | [Matrix factorisation & recommender systems](10_B4_matrix_factorisation_recommenders.md) | Machine learning | 10–12 h |
| 11 | [Graph neural networks: message passing & GCN](11_C3_graph_neural_networks_gcn.md) | Graphs | 12–15 h |
| 12 | [Attention & Graph Attention Networks](12_C4_attention_gat.md) | Graphs | 10–12 h |
| 13 | [Heterogeneous graphs & HAN](13_C5_heterogeneous_graphs_han.md) | Graphs | 8–10 h |
| 14 | [Link prediction](14_C6_link_prediction.md) | Graphs | 10–12 h |
| 15 | [Cheminformatics: SMILES, fingerprints, similarity](15_D2_cheminformatics.md) | Biology / chemistry | 8–10 h |
| 16 | [Ontologies & semantic similarity](16_D3_ontologies_semantic_similarity.md) | Biology / chemistry | 8–10 h |
| 17 | [Biomedical databases & entity resolution](17_D4_biomedical_databases.md) | Biology / chemistry | 6–8 h |
| 18 | [Data leakage & experimental design](18_E1_data_leakage_experimental_design.md) | Research practice | 8–10 h |
| 19 | [Interpretability & its limits](19_E2_interpretability.md) | Research practice | 8–10 h |
| 20 | [Reproducibility & writing up results](20_E3_reproducibility_writing.md) | Research practice | 8–10 h |

**Total: roughly 180–220 hours.** That is about 10–12 weeks at 15–20 hours a
week, or one university semester.

### Shortcuts

- **Already comfortable with Python and ML?** Skim 01–04 (do the exercises to
  check yourself), then start properly at 05.
- **Need to understand the model quickly?** Do 08 → 09 → 11 → 12 → 13 → 14
  first, then fill in the rest.
- **Writing the paper soon?** Read 05, 18, 19 and 20 first: they cover
  evaluation, leakage, interpretability and how to write it all up.

## How the chapters depend on each other

```
01 Python ──► 02 Linear algebra ──► 03 Probability ──► 04 Supervised learning ──► 05 Evaluation
                    │                                          │
                    │                                          ▼
                    │                               07 Neural nets & PyTorch
                    ▼                                          │
              08 Graph theory ──► 09 Random walks ──► 10 Matrix factorisation
                    │                                          │
                    └──────────► 11 GNNs / GCN ◄───────────────┘
                                       │
                                       ▼
                                12 Attention / GAT ──► 13 Heterogeneous / HAN ──► 14 Link prediction

06 Pharmacology ──► 15 Cheminformatics ──► 16 Ontologies ──► 17 Databases

05 + 14 ──► 18 Leakage & design ──► 19 Interpretability ──► 20 Reproducibility & writing
```

## How to study each chapter

1. Read the **motivation** and **objectives** first, so you know what to look for.
2. Read the theory **with a pen**: redo every worked example by hand.
3. **Type** the code examples yourself rather than copy-pasting, run them, and
   change something to see what happens. Use the project environment:
   ```powershell
   cd "C:\Users\Abhineet Anand\Desktop\DrugRepositioning"
   .venv\Scripts\activate
   python
   ```
4. Open the project files named in **"In this project"** and read them alongside.
5. Do the **exercises before** opening the solutions.
6. Answer the **self-check questions** from memory. If you can't, re-read that section.
7. Keep the **cheat sheet** for revision.

## Reading the files

These are Markdown files. In VS Code, open one and press **Ctrl+Shift+V** for
the formatted view (tables, maths, collapsible solutions). Maths written as
`$...$` renders in VS Code's preview and on GitHub.
