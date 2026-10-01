# Unit D2 — Cheminformatics: molecules as graphs, SMILES, fingerprints and similarity

> Part of the self-contained course that accompanies the MV-HGAT drug-repositioning project.
> Track D (biology / chemistry), unit 2 of 4. Comes after D1 (pharmacology and drug repositioning) and before D3 (ontologies).

---

## 0. Front matter

**Prerequisites**

| Unit | What you need from it |
|---|---|
| A1 Python / NumPy / pandas | reading CSVs, boolean masks, `np.triu_indices`, dictionaries and sets |
| A3 Probability | expectations, the "birthday problem" style of counting (used for bit collisions) |
| C1 Graph theory basics | vertices, edges, degree, cycles, neighbourhoods, graph isomorphism (the idea only) |
| D1 Pharmacology | what a drug, a target and an indication are; small molecules versus biologics |

No chemistry beyond secondary school is assumed. Every chemical idea that the project depends on is introduced from scratch.

**Estimated study time:** 12–16 hours (about 6 h reading, 4 h running and modifying the code, 4–6 h of exercises).

**Software:** the project virtual environment (`.venv`) already contains RDKit 2026.03, NumPy, pandas and SciPy. All code in this chapter runs on the CPU in a few seconds. Run the code blocks *in order, in one Python session* (a Jupyter notebook or an IPython console started in the project root `DrugRepositioning/`), because later blocks reuse names defined in earlier ones — exactly like a notebook.

**Learning objectives.** After this unit you will be able to:

1. Describe a molecule as a labelled graph and state, for any atom, its element, number of attached hydrogens, formal charge, aromaticity and ring membership.
2. Read and write SMILES by hand: decode a SMILES string atom by atom (including branches, ring closures, aromatic lower-case atoms, bracket atoms with charges, isotopes and hydrogens, and the stereo marks `@`, `@@`, `/`, `\`), and write valid SMILES for small molecules.
3. Explain the difference between canonical and isomeric SMILES, InChI and InChIKey, and justify why the InChIKey is a good (but not perfect) key for joining chemical databases.
4. Standardise a drug record: recognise salts, counter-ions and mixtures, apply largest-fragment selection and neutralisation, and predict where these rules go wrong (metal complexes, multi-component drugs).
5. Execute the Morgan/ECFP algorithm by hand on a small molecule, explain radius versus diameter (ECFP4 = radius 2), folding, and compute the expected number of bit collisions.
6. Compare structural keys (MACCS), path-based fingerprints (Daylight, RDKit, CDK) and circular fingerprints (ECFP/FCFP), and explain from first principles why the project's ECFP view outperforms the benchmark's CDK view.
7. Compute Tanimoto, Dice, cosine and Tversky coefficients from bit vectors by hand and in code, prove that Dice and Tanimoto rank neighbours identically, and interpret similarity values against fingerprint-specific background distributions.
8. Compute and interpret MW, logP, TPSA, HBD/HBA and the Lipinski rule of five.
9. Explain why biologics (peptides, proteins, antibodies, heparins) have no meaningful SMILES-based similarity and how the project's model copes with the missing values.
10. Read `similarity.py::morgan_tanimoto` and the PubChem part of `02_build_features.py` line by line and modify them confidently.

---

## 1. Motivation: why a drug-repositioning GNN needs chemistry

The model in this project predicts new drug–disease links (indications) on two benchmarks: **Fdataset** (593 drugs × 313 diseases, 1,933 known links) and **Cdataset** (663 drugs × 409 diseases, 2,532 links). Its central bet is *guilt by association*: if drug $r_1$ treats disease $d$, and drug $r_2$ is "similar" to $r_1$, then $r_2$ is a candidate for $d$. Everything therefore hinges on what "similar drugs" means. The project offers the model three drug–drug *views*: two built from chemical structure (`chem_cdk`, `chem_ecfp`) and one from shared genes (`gene_r`).

A concrete example from the data. Three drugs in Fdataset are all classic antidepressants:

| Drug | DrugBank ID | PubChem SMILES (from `data/interim/drugs_all.csv`) |
|---|---|---|
| Imipramine | DB00458 | `CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31` |
| Desipramine | DB01151 | `CNCCCN1C2=CC=CC=C2CCC3=CC=CC=C31` |
| Amitriptyline | DB00321 | `CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31` |

Desipramine is literally imipramine with one methyl group (–CH₃) removed (it is imipramine's main metabolite in the body). Amitriptyline swaps the ring nitrogen for a carbon joined by a double bond to the side chain. A chemist would say all three are "tricyclic" and closely related; a good similarity measure should agree, *and* should still be able to tell them apart.

The benchmark's own matrix (`chem_cdk`, computed in 2011 with the Chemistry Development Kit) says imipramine and desipramine are **identical** (similarity 1.000). Our RDKit ECFP view says 0.611 — similar, but distinguishable. By the end of this chapter you will be able to prove *why* the CDK fingerprint cannot see the difference (Section 6.3) and why that matters.

It matters in the numbers too. Scoring drug–disease pairs using only the 10 nearest neighbours in one view (no learning at all), the project measured (see `docs/HOW_IT_WORKS.md`, Section 10):

| View alone (warm start, drug side) | Fdataset AUPR | Cdataset AUPR |
|---|---|---|
| `chem_cdk` (benchmark, CDK hashed path fingerprint) | 0.061 | 0.047 |
| `chem_ecfp` (ours, RDKit Morgan radius 2, 2,048 bits) | **0.091** | **0.068** |

That is a relative improvement of about 49% (F) and 45% (C) from changing nothing but the fingerprint. Getting structures was itself a small project: the DrugBank XML was unavailable, so the drugs' SMILES were fetched from **PubChem**, where DrugBank deposits its records. 97.5% (F) and 98.9% (C) of drugs received a structure; the rest are mostly biologics such as heparin and calcitonin, which have no SMILES-based similarity at all (Section 9).

So this unit answers four practical questions for the project:

1. How is a molecule written down (SMILES, InChI) and how do we make sure two databases talk about the same molecule (InChIKey, standardisation)?
2. How is a molecule turned into a vector (fingerprints)?
3. How are two vectors compared (Tanimoto and friends), and what is a "high" value?
4. Where does all this break (salts, stereo, metals, biologics, activity cliffs)?

---

## 2. Molecules as graphs

### 2.1 Intuition

A molecule is a set of atoms held together by chemical bonds. If you forget about 3-D shape for a moment, a molecule is exactly a **graph**: atoms are vertices, bonds are edges. The graph is *labelled*: each vertex carries an element (carbon, nitrogen, …) and a few other properties, and each edge carries a bond type (single, double, triple, aromatic). Cheminformatics is, to a first approximation, the study of algorithms on these labelled graphs — which is why a graph-neural-network researcher can learn it quickly.

### 2.2 Atoms

* **Element and atomic number $Z$.** Hydrogen H ($Z=1$), carbon C (6), nitrogen N (7), oxygen O (8), fluorine F (9), phosphorus P (15), sulfur S (16), chlorine Cl (17), bromine Br (35), iodine I (53). Drugs are mostly C, H, N, O with some S, halogens and P; metals (Pt, Au, Fe, Li, Na, Mg, Al) occur in a handful of drugs and salts.
* **Heavy atoms versus hydrogens.** Hydrogens are so numerous and so predictable that molecular graphs usually leave them out. A carbon written alone is understood to carry enough hydrogens to fill its valence (Section 2.4). These are **implicit hydrogens**; the graph has only **heavy atoms** (non-hydrogen) as vertices. Aspirin, C₉H₈O₄, has 21 atoms but only 13 heavy atoms, and its graph has 13 vertices.
* **Isotope.** Atoms of the same element can have different masses (number of neutrons). Normally we ignore this, but radiopharmaceuticals depend on it: the project contains **ioflupane I-123** (DB08824), whose SMILES contains `[123I]`, an iodine-123 atom used for SPECT imaging of dopamine transporters.
* **Formal charge.** A bookkeeping charge assigned to an atom when it has more or fewer bonds than usual: ammonium N in NH₄⁺ has charge +1; a carboxylate O in –COO⁻ has −1. Many drugs are stored as salts and carry charged atoms (Section 5).

### 2.3 Bonds

| Bond | Order | SMILES symbol | Example |
|---|---|---|---|
| single | 1 | `-` (usually omitted) | C–C in ethane `CC` |
| double | 2 | `=` | C=O in formaldehyde `C=O` |
| triple | 3 | `#` | C≡N in a nitrile `C#N` (citalopram has one) |
| aromatic | "1.5" | `:` (usually omitted) | bonds in benzene `c1ccccc1` |

### 2.4 Valence: how many bonds an atom wants

Each element has a small set of normal **valences** — the total bond order it likes to have, counting hydrogens:

| Element | Normal valence(s) | So a lone atom becomes |
|---|---|---|
| C | 4 | CH₄ |
| N | 3 (5 in e.g. nitro groups written without charges) | NH₃ |
| O | 2 | H₂O |
| S | 2, 4, 6 | H₂S |
| P | 3, 5 | PH₃ |
| F, Cl, Br, I | 1 | HF, HCl, … |

The **implicit-hydrogen rule** used by SMILES for the common "organic subset" atoms is: count the bond orders the atom already has, then add hydrogens up to the *lowest normal valence that is at least that count*. For carbon in `C=O`, the carbon already has bond order 2, so it gets 2 hydrogens (formaldehyde, H₂C=O). A carbon in `CC(C)(C)C` with four carbon neighbours gets zero. A carbon with five bonds is impossible, and a toolkit will refuse the molecule (Section 3.10).

Formal charge shifts valence: N⁺ behaves like carbon (4 bonds, as in ammonium), O⁻ like a halogen (1 bond).

### 2.5 Rings and aromaticity

A **ring** is a cycle in the graph. The number of independent rings is the graph's **cyclomatic number** $\mu = |E| - |V| + c$, where $c$ is the number of connected components. For amantadine (11 heavy atoms, 13 bonds, one component), $\mu = 13 - 11 + 1 = 3$: its cage has three independent rings.

**Aromaticity** is a special stability of certain flat rings with alternating single and double bonds. Benzene (C₆H₆) can be drawn with alternating bonds in two ways (two **Kekulé structures**); in reality all six bonds are identical, intermediate between single and double. Hückel's rule is the classic test: a flat, fully conjugated ring with $4n+2$ π-electrons ($n = 0, 1, 2, \ldots$: 2, 6, 10, … electrons) is aromatic. Benzene has 6 (n = 1). Pyridine (a nitrogen in place of one CH) and pyrrole (a five-membered ring with an N–H) are aromatic too.

Two consequences matter in practice:

1. *Toolkits disagree on aromaticity.* RDKit, CDK, OpenEye and Daylight each implement their own aromaticity model (which rings, which heteroatoms, fused systems). The same molecule can therefore get slightly different fingerprints in different toolkits even with identical algorithms. This is one reason we never mix fingerprints from different toolkits in one similarity matrix.
2. *Aromatic and Kekulé input must give the same molecule.* PubChem writes SMILES in Kekulé form (`C1=CC=CC=C1`); RDKit perceives aromaticity on reading and writes `c1ccccc1`. Both describe benzene.

### 2.6 Stereochemistry, briefly

Two molecules can have the same atoms and the same bonds — the same graph — yet differ in 3-D arrangement. These are **stereoisomers**. The two kinds you meet in drug data are:

* **Tetrahedral (chiral) centres.** A carbon with four *different* substituents can be arranged in two mirror-image ways, like left and right hands. The two mirror images are **enantiomers**; they are labelled *R* or *S* by the Cahn–Ingold–Prelog (CIP) priority rules. Molecules with several centres can also form **diastereomers** (stereoisomers that are not mirror images).
* **Double-bond (cis/trans, E/Z) isomerism.** Rotation about a C=C is blocked, so substituents can sit on the same side (*Z*, "cis") or opposite sides (*E*, "trans").

Stereo matters biologically because protein binding sites are themselves chiral. Pairs in this project's drug list:

| Pair | Relationship | Clinical note |
|---|---|---|
| Citalopram / escitalopram | racemate (no stereo specified) / pure *S* enantiomer | both antidepressants; the *S* form carries the activity |
| Quinine / quinidine | diastereomers (differ at two centres) | quinine: antimalarial; quinidine: antiarrhythmic (also antimalarial) |
| Betamethasone / dexamethasone | epimers (differ at one centre) | both corticosteroids |
| Omeprazole / esomeprazole | racemate / *S* enantiomer | both proton-pump inhibitors |

Most 2-D fingerprints ignore stereo by default — a design decision with consequences (Section 6.6).

### 2.7 The formal picture

A molecule is a labelled, undirected, simple graph $G = (V, E, \lambda_V, \lambda_E)$ with

* $V$ = heavy atoms,
* $E \subseteq \binom{V}{2}$ = bonds,
* $\lambda_V(v)$ = a tuple of atom labels, e.g. (atomic number, formal charge, implicit-H count, isotope, aromatic flag, chirality tag),
* $\lambda_E(e)$ = bond type (single/double/triple/aromatic) plus a stereo flag for double bonds.

Two molecules are the same compound if their labelled graphs are **isomorphic** (there is a label-preserving bijection between atoms that preserves bonds). Testing graph isomorphism is not known to be polynomial-time in general, and comparing *how similar* two graphs are (maximum common subgraph) is NP-hard. This is the fundamental reason cheminformatics converts molecules to strings (for identity) and fixed-length vectors (for similarity).

Here is aspirin (acetylsalicylic acid, DB00945) with RDKit's atom numbering, which follows the order of atoms in its SMILES `CC(=O)OC1=CC=CC=C1C(=O)O`:

```
                                      O2
                                      ║
                 H                    ║
                 |                    ║
            H ─ C5 ══ C4 ── O3 ──── C1 ──── C0H3      (acetyl ester part)
               /        \
          H ─ C6         C9 ──── C10 ──── O12 ─ H    (carboxylic acid part)
               \\       //        ║
          H ─  C7 ── C8           O11
                      |
                      H
     ══, //, \\ : double bonds of one Kekulé structure (ring is aromatic)
```

The vertex labels are: C0 (carbon, 3 H), C1 (carbon, 0 H), O2 (oxygen, double-bonded), O3 (ester oxygen), C4–C9 (aromatic ring carbons; C5–C8 each carry one H), C10 (carboxyl carbon), O11 (C=O oxygen), O12 (hydroxyl oxygen, 1 H).

---

## 3. SMILES in depth

### 3.1 The idea

SMILES (Simplified Molecular-Input Line-Entry System; Weininger, 1988) writes a molecular graph as a line of text by walking the graph **depth-first** from some starting atom. Atoms are written as they are visited; bonds between consecutive atoms are implied; side branches are put in parentheses; and every time the walk would need to "jump back" to close a ring, a matching pair of digits marks the two ends of the ring bond. The grammar is small enough to learn in an afternoon. The modern, vendor-neutral specification is **OpenSMILES**; the original reference is the Daylight *SMILES Theory Manual* (links in Section 17).

### 3.2 Atoms

**Organic subset (no brackets):** `B C N O P S F Cl Br I` and, for aromatic atoms, lower-case `b c n o p s`. These atoms get implicit hydrogens by the valence rule of Section 2.4.

**Bracket atoms:** anything else, or anything with a non-default property, is written in square brackets with the fields in a fixed order:

```
[ isotope  symbol  chirality  hcount  charge  :class ]
    13       C        @@         H3      +1
```

Inside brackets **no implicit hydrogens are added** — you must write them. Examples:

| SMILES | Meaning |
|---|---|
| `[Na+]` | sodium cation |
| `[O-]` | oxygen with charge −1 and no hydrogens (e.g. in a carboxylate) |
| `[NH4+]` | ammonium |
| `[13CH4]` | methane containing carbon-13 |
| `[123I]` | iodine-123 (ioflupane) |
| `[nH]` | aromatic nitrogen that carries a hydrogen (pyrrole, indole, imidazole) |
| `[Pt+2]` | platinum(II) cation (cisplatin, carboplatin as written by PubChem) |
| `[C@@H]` | a chiral carbon with one hydrogen (Section 3.8) |
| `[Fe+4]`, `[Li+]`, `[Mg+2]`, `[Au]` | metals (sodium nitroprusside, lithium, magnesium sulfate, auranofin) |

### 3.3 Bonds and disconnection

`-` single, `=` double, `#` triple, `$` quadruple (rare), `:` aromatic. Single and aromatic bonds are normally omitted: between two aliphatic atoms an omitted bond is single; between two aromatic atoms it is aromatic.

The dot `.` means **no bond**: the string describes several disconnected pieces (fragments). `CCO.O` is ethanol plus a water molecule. Salts are written this way: `CN(C)CCCN1c2ccccc2CCc2ccccc21.Cl` is imipramine hydrochloride. Dots are the reason the project needs a standardisation step (Section 5).

### 3.4 Branches

Parentheses hold a side branch; the walk continues after the closing parenthesis from the atom *before* the opening parenthesis. `CC(=O)O` is acetic acid: C, then C, which has a branch `=O`, then the main chain continues from that second C to `O`. Branches can be nested and stacked: `CC(C)(C)C` is neopentane (the central carbon has four carbon neighbours).

### 3.5 Rings

To write a ring, break one ring bond, write the resulting chain, and mark the two atoms of the broken bond with the same digit. `C1CCCCC1` is cyclohexane: the first and last carbons carry the digit 1, which means "these two are bonded". A bond symbol may be put before the digit to give the ring bond a type (`C1=CCCCC1`). Digits can be reused once the ring is closed; ring-closure numbers above 9 are written `%10`, `%11`, …. One atom can carry several digits (`C12` opens or closes two rings at once).

### 3.6 Aromatic atoms

Lower-case atoms are aromatic: `c1ccccc1` is benzene, `c1ccncc1` pyridine, `c1cc[nH]c1` pyrrole, `c1ccoc1` furan, `c1ccsc1` thiophene. A toolkit must be able to assign a valid alternating single/double pattern to an aromatic system (**kekulisation**). If it cannot — as for `c1cccc1`, a five-carbon "aromatic" ring with no heteroatom — the SMILES is rejected.

The pyrrole case shows why `[nH]` exists. A five-membered ring has an odd number of atoms, so its atoms cannot all be paired up by alternating double bonds; one atom must instead contribute a lone pair of electrons to the aromatic system. In pyrrole that atom is the nitrogen, and it carries a hydrogen. Written as `c1ccnc1`, the parser does not know this and kekulisation fails; `c1cc[nH]c1` says explicitly which atom is the N–H.

### 3.7 Charges and the two ways to write a nitro group

Charges go inside brackets: `[N+]`, `[O-]`, `[Fe+4]`, `[Pt+2]` (`++` is an old synonym for `+2`). Some groups can be written with or without charge separation. The nitro group –NO₂ is properly `[N+](=O)[O-]` (nitrogen with four bonds and a positive charge, one oxygen negative); some databases write the "pentavalent" form `N(=O)=O`. Both describe the same thing, and a standardiser converts one into the other so that fingerprints agree.

### 3.8 Stereochemistry in SMILES

**Tetrahedral centres: `@` and `@@`.** Look from the *first* neighbour of the chiral atom (the atom written before it) towards the chiral atom. The remaining neighbours, *in the order they are written*, appear either anticlockwise (`@`, think of the spiral of the @ sign) or clockwise (`@@`). An implicit hydrogen written inside the bracket (`[C@@H]`) counts as the neighbour immediately following the "from" atom. Ring-closure digits count at the position where the digit appears.

*Worked example: L-alanine.* `N[C@@H](C)C(=O)O`. The chiral carbon's neighbours in written order are N, H, CH₃, COOH, and `@@` says: looking from N, the sequence H → CH₃ → COOH is clockwise. To get the CIP label we must look with the lowest-priority group (H) pointing away. Two facts make this mechanical: swapping any two neighbours in the list flips `@` ↔ `@@`, and looking *from* an atom versus looking with that atom *behind* reverses the sense of rotation.

* Swap N and H: (H, N, CH₃, COOH) with `@` (anticlockwise).
* Swap CH₃ and COOH: (H, N, COOH, CH₃) with `@@` (clockwise).
* So looking *from* H, the order N → COOH → CH₃ is clockwise; with H *behind* the centre, it is anticlockwise.
* CIP priorities are N (atomic number 7) > COOH > CH₃ > H, so N → COOH → CH₃ is the priority order 1 → 2 → 3, and anticlockwise means ***S***. L-alanine is indeed (*S*)-alanine, and RDKit agrees (Section 11).

The project's rivastigmine, `CCN(C)C(=O)OC1=CC=CC(=C1)[C@H](C)N(C)C`, is the (*S*) enantiomer, and escitalopram's `[C@@]` likewise encodes (*S*).

**Double bonds: `/` and `\`.** These are "directional single bonds" that tell you on which side of the double bond a substituent is. Read them as little arrows of the bond's direction relative to the line of text: `F/C=C/F` has both fluorines "up then up" relative to the direction of writing, which places them on *opposite* sides — *trans* (*E*)-1,2-difluoroethene. `F/C=C\F` puts them on the *same* side — *cis* (*Z*). A handy mnemonic: the same symbol on both sides gives *trans*, different symbols give *cis* (when both are written after their atoms as here).

### 3.9 Worked example 1 — decoding aspirin atom by atom

`CC(=O)OC1=CC=CC=C1C(=O)O` (as written by PubChem, Kekulé form)

| Step | Token | Action | Atom created / bond added |
|---|---|---|---|
| 1 | `C` | first atom | atom 0: C |
| 2 | `C` | new atom, single bond to previous | atom 1: C; bond 0–1 |
| 3 | `(` | open branch at atom 1 | |
| 4 | `=O` | double-bonded O | atom 2: O; bond 1=2 |
| 5 | `)` | close branch, return to atom 1 | |
| 6 | `O` | single bond to atom 1 | atom 3: O; bond 1–3 |
| 7 | `C1` | C, single bond to 3, **ring-bond label 1 opened** | atom 4: C; bond 3–4 |
| 8 | `=C` | double bond | atom 5; bond 4=5 |
| 9 | `C` | | atom 6; bond 5–6 |
| 10 | `=C` | | atom 7; bond 6=7 |
| 11 | `C` | | atom 8; bond 7–8 |
| 12 | `=C1` | double bond, and **ring label 1 closed** | atom 9; bond 8=9; ring bond 9–4 |
| 13 | `C` | | atom 10; bond 9–10 |
| 14 | `(=O)` | branch | atom 11; bond 10=11 |
| 15 | `O` | | atom 12; bond 10–12 |

Result: 13 heavy atoms, 13 bonds, one ring ($\mu = 13 - 13 + 1 = 1$). Now the implicit hydrogens by the valence rule: C0 has bond order 1 → 3 H; C1 has 1 + 2 + 1 = 4 → 0 H; O2 has 2 → 0 H; O3 has 2 → 0 H; ring carbons C5–C8 each have 1 + 2 = 3 → 1 H each; C4 and C9 have 4 → 0 H; C10 has 4 → 0 H; O11 → 0 H; O12 has 1 → 1 H. Total hydrogens 3 + 4 + 1 = 8, so the formula is C₉H₈O₄ — aspirin. The six-membered ring with alternating double bonds is aromatic, so RDKit writes it canonically as `CC(=O)Oc1ccccc1C(=O)O`.

### 3.10 Worked example 2 — the amantadine cage

Amantadine (DB00915), an antiviral later repositioned for Parkinson's disease, is `C1C2CC3CC1CC(C2)(C3)N`.

Walk: atom 0 `C1` (opens ring 1); atom 1 `C2` (opens ring 2; bond 0–1); atom 2 `C` (1–2); atom 3 `C3` (opens ring 3; 2–3); atom 4 `C` (3–4); atom 5 `C1` (4–5, **closes ring 1: bond 5–0**); atom 6 `C` (5–6); atom 7 `C` (6–7); branch `(C2)`: atom 8 (7–8, **closes ring 2: bond 8–1**); branch `(C3)`: atom 9 (7–9, **closes ring 3: bond 9–3**); atom 10 `N` (7–10).

Bonds: 0–1, 1–2, 2–3, 3–4, 4–5, 5–0, 5–6, 6–7, 7–8, 8–1, 7–9, 9–3, 7–10 → 13 bonds on 11 atoms, so three independent rings — the adamantane cage. Hydrogens: atoms 0, 2, 4, 6, 8, 9 have two carbon neighbours → CH₂ (12 H); atoms 1, 3, 5 have three → CH (3 H); atom 7 has four heavy neighbours → no H; N has one bond → NH₂ (2 H). Total C₁₀H₁₇N. Memantine (DB01043, an Alzheimer drug) is the same cage with two extra methyl groups: C₁₂H₂₁N.

### 3.11 Worked example 3 — imipramine's tricycle

`CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31`

* `CN(C)` — a nitrogen with two methyl groups (dimethylamino).
* `CCC` — a three-carbon chain.
* `N1` — the ring nitrogen; opens ring label 1.
* `C2=CC=CC=C2` — a benzene ring (label 2 opened and closed), fused because its first carbon is bonded to N1 and its last carbon continues the walk.
* `CC` — an ethylene bridge (–CH₂–CH₂–).
* `C3=CC=CC=C31` — a second benzene ring; its last carbon closes label 3 (the benzene ring) **and** label 1 (back to the nitrogen), creating the central seven-membered ring.

So imipramine is two benzene rings joined by an N on one side and a CH₂–CH₂ on the other (a dibenzazepine), with a dimethylaminopropyl tail on the nitrogen. Desipramine is `CNCCC...`: the same with one methyl on the tail nitrogen.

### 3.12 Canonical and isomeric SMILES

A molecule with $n$ atoms can be written as many different SMILES as there are depth-first walks (start atom × neighbour orderings): `OCC`, `C(O)C` and `CCO` are all ethanol. For **identity** we want one string per molecule: the **canonical SMILES**. Canonicalisation works in two stages:

1. **Rank the atoms** with a graph-invariant algorithm. The classic is Morgan's (1965) "extended connectivity" iteration — the same idea as the ECFP fingerprint in Section 6.4: start with atom invariants, repeatedly refine each atom's rank by its neighbours' ranks until the ranking stops changing, then break remaining ties (symmetric atoms) deterministically.
2. **Write the SMILES** starting at the lowest-ranked atom and always visiting neighbours in rank order.

Two practical warnings:

* Canonical SMILES are canonical **only within one toolkit and version.** RDKit, Open Babel, OpenEye, CDK and PubChem each produce their own canonical form. Never compare canonical SMILES produced by different software; compare InChIKeys or re-canonicalise everything with one toolkit.
* **Isomeric SMILES** include stereo (`@`, `@@`, `/`, `\`) and isotopes (`[13C]`); **non-isomeric** SMILES strip them. PubChem renamed its properties in 2025: the property `SMILES` is now the full (isomeric) form and `ConnectivitySMILES` the connectivity-only form (the old names `IsomericSMILES` and `CanonicalSMILES` remain as aliases). The project requests `SMILES`, which is why escitalopram arrives with its `[C@@]`.

### 3.13 Invalid SMILES and common traps

* **Valence errors.** `C(C)(C)(C)(C)C` asks a carbon to have five bonds; RDKit returns `None`.
* **Unkekulisable aromatics.** `c1cccc1` cannot be given alternating bonds; RDKit returns `None`. Pyrrole needs `[nH]`.
* **Hydrogens inside brackets.** `[C]` is a bare carbon atom with *no* hydrogens (a radical), not methane. Methane is `C` or `[CH4]`.
* **Silent dots.** `CCO.Cl` is two molecules. A fingerprint computed on the whole string mixes features of the salt into the drug.
* **Always check for `None`.** `Chem.MolFromSmiles` returns `None` instead of raising an exception; the project's code checks for it explicitly (Section 12).

---

## 4. InChI and InChIKey: identifiers for joining databases

### 4.1 InChI

The **International Chemical Identifier** (InChI), developed by IUPAC and the InChI Trust, is a *canonical identifier by design*: one algorithm, one open-source reference implementation, so any software produces the same InChI for the same structure. It is built from **layers**, each prefixed by a letter, so that the string reads from general to specific. Aspirin:

```
InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)
       │   │      │                               │
       │   │      │                               └ /h hydrogen layer: one H on atoms 2-5, three on atom 1,
       │   │      │                                  and a "mobile" H shared by atoms 11 and 12 (the acid O's)
       │   │      └ /c connection layer: the skeleton, using InChI's own canonical atom numbers
       │   └ formula layer
       └ version 1, S = Standard InChI (fixed options so everyone gets the same string)
```

Further optional layers: `/q` (charge), `/p` (protons added or removed), `/b` (double-bond stereo), `/t` (tetrahedral stereo), `/m` and `/s` (whether the stereo is absolute or relative), `/i` (isotopes). Rivastigmine's InChI ends with `/t11-/m0/s1`: atom 11 is a stereocentre with parity "−", and the stereo is absolute.

The mobile-hydrogen notation `(H,11,12)` is one of InChI's strengths: it represents a carboxylic acid whichever oxygen the hydrogen is drawn on, and it partly normalises **tautomers** (structures that differ only by where a hydrogen sits), which canonical SMILES do not.

### 4.2 InChIKey

An InChI can be hundreds of characters long, which makes it awkward as a database key or search term. The **InChIKey** is a fixed-length, 27-character *hash* of the InChI:

```
BSYNRYMUTXBXSQ - UHFFFAOYSA - N
└──── 14 ────┘   └── 10 ──┘  └ 1: protonation flag (N = neutral, M = one proton removed, O = one added, ...)
 hash of the       8 chars: hash of the remaining layers (stereo, isotopes, ...)
 skeleton          + 'S' (standard) + 'A' (InChI version 1)
 (connectivity)
```

Facts you can read straight off a key:

* `UHFFFAOYSA` is the hash of "nothing" in the second block: the molecule has **no stereo or isotope layer** (aspirin, imipramine, amantadine).
* **Stereoisomers share the first block.** From the project:

| Pair | InChIKeys |
|---|---|
| citalopram / escitalopram | `WSEQXVZVJXJVFP-UHFFFAOYSA-N` / `WSEQXVZVJXJVFP-FQEVSTJZSA-N` |
| quinine / quinidine | `LOUPRKONTZGTKE-WZBLMQSHSA-N` / `LOUPRKONTZGTKE-LHHVKLHASA-N` |
| betamethasone / dexamethasone | `UREBDLICKHMUKA-DVTGEIKXSA-N` / `UREBDLICKHMUKA-CXSFZGCWSA-N` |

So "same first block" is a quick test for "same skeleton, possibly different stereo".
* **Charge shows in the last character.** Sodium tetradecyl sulfate as a sodium salt has key `…-UHFFFAOYSA-M`; ammonium `[NH4+]` ends in `-O`.
* **Salts change the key completely.** Imipramine is `BCGWQEUPMDMJNV-UHFFFAOYSA-N`; imipramine hydrochloride (InChI `…/C19H24N2.ClH/…`) is `XZZXIYZZBJDEEP-UHFFFAOYSA-N`. The first block differs because the hash covers *all* components. Standardise first, then compute keys.

### 4.3 Why the InChIKey is a good join key

Joining databases (DrugBank → PubChem → CTD) is the unglamorous step every repositioning project depends on. Candidate keys and their problems:

| Key | Problem |
|---|---|
| drug name | synonyms, brand names, salts ("imipramine hydrochloride"), spelling, language |
| CAS number | proprietary; different numbers for salts and hydrates; not always present |
| database IDs (DrugBank, PubChem CID, ChEMBL) | each database has its own; cross-references can be missing or stale |
| canonical SMILES | toolkit- and version-dependent (Section 3.12) |
| **InChIKey** | computed *from the structure* by one open algorithm; fixed length; no special characters; indexed by search engines; present in PubChem, ChEMBL, CTD, ChEBI, UniChem |

Why trust a 14-character hash not to collide? The first block encodes about 65 bits of a SHA-256 hash. By the birthday bound, the chance that *any* two of $N$ distinct skeletons share a first block is about $N^2 / 2^{66}$; for the roughly $10^8$ compounds in PubChem this is about $10^{16} / 7.4 \times 10^{19} \approx 1.4 \times 10^{-4}$. Pletnev et al. (2012) tested the hash experimentally on very large structure sets and found it behaves like an ideal hash, so for a few thousand drugs a collision is not a practical concern. When absolute certainty is required, keep the full InChI next to the key.

Limitations to keep in mind:

* **Salts and mixtures** (above): standardise first.
* **Stereo:** a racemate and an enantiomer have different full keys; use the first block if you want them joined.
* **Tautomers:** standard InChI handles common mobile-hydrogen cases but not all tautomerism.
* **Biologics and polymers:** no single structure, so no InChI. Heparin, antibodies and glatiramer cannot be joined this way.
* **Organometallics:** standard InChI disconnects metals from organic ligands, so metal complexes are represented imperfectly.

In this project, the drug → CTD join (`02_build_features.py::ctd_chemical_map`) uses a cascade: PubChem CID first, then InChIKey, then name — exactly the robustness ordering suggested by the table above.

---

## 5. Salts, counter-ions, mixtures and standardisation

### 5.1 Why drug records contain more than one molecule

Many drugs are sold as **salts**: the active molecule (often a basic amine or a carboxylic acid) is paired with a **counter-ion** of opposite charge — chloride (hydrochlorides), sodium, sulfate, maleate, mesylate, and so on. Salts dissolve better, crystallise more reliably and keep longer. A registration record therefore often describes "imipramine hydrochloride", not imipramine. Other records are genuine **mixtures** (two active components), **hydrates** (with water molecules in the crystal) or **co-crystals**.

In SMILES all of these appear as dot-separated fragments. Real cases from this project's `drugs_all.csv`:

| Drug | PubChem SMILES (abridged) | What the fragments are |
|---|---|---|
| Aminophylline (DB01223) | `CN1C2=C(C(=O)N(C1=O)C)NC=N2.CN1C2=C(...)NC=N2.C(CN)N` | two theophylline molecules + one ethylenediamine (a 2:1 salt that improves solubility) |
| Sodium tetradecyl sulfate (DB00464) | `CCCCC(CC)CCC(CC(C)C)OS(=O)(=O)[O-].[Na+]` | the sulfate anion + sodium |
| Sodium bicarbonate (DB01390) | `C(=O)(O)[O-].[Na+]` | bicarbonate + sodium |
| Carboplatin (DB00958) | `C1CC(C1)(C(=O)O)C(=O)O.N.N.[Pt+2]` | the platinum complex written as *disconnected* ligands and metal |
| Cisplatin (DB00515) | `N.N.Cl[Pt+2]Cl` | two ammonia ligands + PtCl₂ |
| Pituitrin (DB00067) | two long peptide SMILES joined by `.` | a mixture of two hormones (vasopressin and oxytocin) |

Overall, 10 of the 578 Fdataset drugs with a structure (and 11 of 656 in Cdataset) have multi-fragment SMILES.

### 5.2 Why fragments must be removed before fingerprinting

A fingerprint is computed over *all* atoms of the molecule object. If the counter-ion is left in:

* The salt's features are added to the drug's bit vector. Two different hydrochloride salts share the "Cl" bits, which inflates their similarity; a drug and its own salt look less similar than they should.
* Aminophylline and theophylline are *pharmacologically the same drug* (aminophylline releases theophylline in the body), but their raw-SMILES ECFP Tanimoto is only 0.839, because ethylenediamine adds extra bits. After keeping the largest fragment, it is exactly 1.0 (verified in Section 11).
* InChIKeys of the salt and the parent differ (Section 4.2), so database joins fail.

### 5.3 A standardisation pipeline

"Standardisation" means mapping every record to one agreed **parent** representation. A typical pipeline (RDKit's `rdMolStandardize`, the ChEMBL structure pipeline and PubChem's standardiser all implement variants) is:

1. **Sanitise / clean up**: parse, check valences, perceive aromaticity, fix known drawing problems (e.g. convert pentavalent nitro `N(=O)=O` to the charge-separated `[N+](=O)[O-]`), disconnect metal–organic bonds in a consistent way.
2. **Choose the parent fragment**: remove salts/solvents by a list of known counter-ions, or simply keep the **largest fragment**.
3. **Neutralise** ("uncharge"): add or remove protons so that charged acids and bases become neutral where that is chemically sensible (`[O-]` → `O` with an H; `[NH3+]` → `N`). Quaternary ammonium ions, which cannot be neutralised, stay charged.
4. **Canonicalise tautomers** (optional; expensive and occasionally surprising).
5. **Handle stereo** consistently (keep it, or drop it if you want racemates and enantiomers merged).
6. **Recompute** canonical SMILES and InChIKey from the parent.

### 5.4 Largest fragment: the exact rule and its failure modes

RDKit's `LargestFragmentChooser` (inherited from the MolVS standardiser) picks the fragment with **the most atoms — counting hydrogens by default** — breaking ties by higher molecular weight and then by alphabetical SMILES. The project applies only this step (no neutralisation) before fingerprinting:

```python
# quoted from src/drepo/similarity.py, morgan_tanimoto
largest = rdMolStandardize.LargestFragmentChooser()
...
mol = largest.choose(mol)   # drop counter-ions such as HCl
```

It works perfectly for ordinary salts: imipramine·HCl → imipramine, aminophylline → theophylline (two identical theophylline fragments with 21 atoms each beat ethylenediamine's 12). It fails for drugs whose activity *is* the combination:

* **Cisplatin** `N.N.Cl[Pt+2]Cl`: each ammonia NH₃ has 4 atoms with its hydrogens, PtCl₂ only 3 → the "largest fragment" is **ammonia**. The project's ECFP for cisplatin is therefore the fingerprint of NH₃ — chemically meaningless.
* **Carboplatin**: the cyclobutanedicarboxylic acid ligand wins; the platinum, which is the whole point of the drug, disappears.
* **Sodium nitroprusside** (`[C-]#N` ×5, `[N-]=O`, `[Fe+4]`): the iron–cyanide–nitrosyl complex is written as seven ions; the cyanide and nitrosyl fragments tie on atom count, molecular weight breaks the tie, and the "drug" becomes a single N=O.
* **Vitamin B12 forms** (hydroxocobalamin, "Anacobin" and "Codroxomin" in our table): PubChem writes the cobalt as a separate ion, so the corrin ring is kept and the cobalt dropped.
* **Mixtures** like Pituitrin keep one hormone and drop the other.

Exercise 8 finds all of them automatically. These are a handful of drugs, so they hardly move the averages, but they are exactly the kind of case a reviewer will ask about. A more careful pipeline would (a) flag records containing metals and keep them whole or use a metal-aware representation, and (b) use a curated salt list (e.g. RDKit's `SaltRemover`) so that only known counter-ions are removed.

**Neutralisation** matters less for fingerprints in this project because ECFP invariants include formal charge only on the charged atom itself; but it matters for InChIKey joins and for descriptors such as logP (an ionised acid looks far more water-loving than the neutral molecule).

---

## 6. Molecular fingerprints

### 6.1 Why vectors?

We need to compare each of ~600 drugs with every other (about 180,000 pairs per benchmark), and in virtual screening one compares a query with millions of molecules. Exact graph comparison (maximum common substructure) is NP-hard. A **fingerprint** turns a molecule into a fixed-length vector — usually of bits — such that *similar structures give overlapping bit patterns*. Comparing two fingerprints is then a handful of CPU instructions (AND, OR, population count).

All fingerprints follow the same recipe: **(1) enumerate structural features** of the molecule; **(2) map each feature to one or more positions** in a vector; **(3) set those positions**. They differ in what a "feature" is.

### 6.2 Structural keys: MACCS

A **structural-key** fingerprint has a predefined dictionary of substructure questions, one bit per question: "is there a ring of size 6?", "is there an S–S bond?", "are there more than 2 oxygens?". The widely used public set is the **MACCS 166 keys** (from MDL; documented by Durant et al., 2002), implemented in RDKit as `MACCSkeys.GenMACCSKeys` (a 167-bit vector whose bit 0 is unused).

* *Pros:* every bit has a human meaning; no collisions.
* *Cons:* only 166 questions, chosen decades ago for substructure search, not for bioactivity. Many molecules share most keys, so MACCS similarities are high even for unrelated molecules (imipramine vs donepezil: 0.54) and the *scale* of "similar" differs from other fingerprints.

### 6.3 Path-based hashed fingerprints: Daylight, RDKit and CDK

A **path-based** fingerprint enumerates all **linear paths** (and, in RDKit's version, also branched subgraphs) of the molecular graph from 0 or 1 up to $L$ bonds. Each path is turned into a string of atom and bond labels, e.g. `C-C-N-C`, read in a canonical direction; the string is **hashed** to an integer, and that integer (modulo the vector length) selects the bit(s) to set. The original Daylight fingerprint enumerated paths of up to 7 bonds and set several bits per path. Typical parameters:

| Implementation | Features | Length | Bits per feature |
|---|---|---|---|
| Daylight | linear paths, 0–7 bonds | 2,048 (often folded further) | several |
| RDKit `RDKFingerprint` | linear **and branched** subgraphs, 1–7 bonds | 2,048 | 2 |
| CDK `Fingerprinter` (used by the benchmark) | linear paths up to depth 8 | 1,024 | 1 |

The **Fdataset/Cdataset drug similarity (`chem_cdk`)** comes from Gottlieb et al. (2011): canonical SMILES from DrugBank, *hashed fingerprints computed with CDK using default parameters*, then Tanimoto. In CDK of that era the defaults were 1,024 bits and a search depth of 8.

**The blind spot of hydrogen-suppressed paths.** Path labels are built from element symbols and bond orders; implicit hydrogens are not part of the label. Removing a terminal methyl group therefore removes only paths whose *label sequences already occur elsewhere* in the molecule whenever an equivalent carbon chain exists. For imipramine and desipramine, every label sequence of the form C–N–C, C–C–C–N–C, … that the extra methyl creates is already generated by the remaining methyl or the chain carbons. The *set* of path labels is identical, so any hashed path fingerprint is identical, so the Tanimoto is exactly 1. Section 11 enumerates the path labels (up to 7 bonds) of both molecules and confirms that the two sets are identical (90 labels each), and the same holds for amantadine versus memantine (the extra methyl groups on the cage create only C–C–…–C sequences that the cage already contains). This is precisely what the benchmark matrix shows: `chem_cdk` = 1.000 for both pairs.

### 6.4 Circular fingerprints: Morgan / ECFP

**Extended-connectivity fingerprints** (ECFP; Rogers & Hahn, 2010) are the de-facto standard for bioactivity modelling. Instead of paths, an ECFP feature is a **circular atom environment**: an atom plus everything within $r$ bonds of it. The algorithm is a direct descendant of Morgan's 1965 canonical-numbering algorithm, which is why RDKit calls it the *Morgan fingerprint*.

**Intuition.** Each atom starts with a code describing itself. Then, in each round, every atom updates its code by hashing together its own code and the codes of its neighbours — so after round 1 the code describes the atom and its immediate neighbours; after round 2, the neighbours' neighbours too. The set of all codes produced in all rounds is the fingerprint. If you know GNNs, this is a message-passing network with a *fixed, non-learned* hash function as the update, and a set (not a sum) as the readout. In fact, it is the Weisfeiler–Lehman graph-isomorphism test run for $R$ iterations.

**Step 0 — initial atom invariants.** For each heavy atom, collect a tuple of properties that do not depend on how the molecule is numbered. Rogers & Hahn use the "Daylight atomic invariants" plus ring membership:

1. number of heavy-atom neighbours (degree),
2. valence minus number of hydrogens (i.e. total bond order to heavy atoms),
3. atomic number,
4. atomic mass (isotope),
5. formal charge,
6. number of attached hydrogens,
7. whether the atom is in at least one ring.

RDKit's default uses an equivalent set: atomic number, total degree (including H), number of hydrogens, formal charge, isotope mass difference, ring membership. The tuple is hashed to a 32-bit integer — the atom's **radius-0 identifier**.

**Step $k$ (= 1, 2, …, R) — neighbourhood update.** For every atom $a$ with current identifier $\mathrm{id}_{k-1}(a)$:

1. List its neighbours as pairs (bond order, $\mathrm{id}_{k-1}(\text{neighbour})$).
2. **Sort** the pairs (so the result does not depend on atom numbering).
3. Hash the tuple $(k,\ \mathrm{id}_{k-1}(a),\ \text{sorted pairs})$ to a new 32-bit integer $\mathrm{id}_k(a)$.

Formally,
$$
\mathrm{id}_k(a) = h\Big(k,\ \mathrm{id}_{k-1}(a),\ \mathrm{sort}\big\{(\beta_{ab},\ \mathrm{id}_{k-1}(b)) : b \in N(a)\big\}\Big),
$$
where $N(a)$ is the set of neighbours, $\beta_{ab}$ the bond order, and $h$ a hash function.

**Duplicate-structure removal.** After each round, record for every atom the **set of bonds** its environment covers. If two environments cover exactly the same bonds (for example, a terminal methyl at radius 2 covers the same three bonds as its neighbouring carbonyl carbon at radius 1), they describe the *same substructure*; keep only one of them (the one from the earlier round or, within a round, a deterministic choice).

**Collect.** The fingerprint is the *set* of all distinct identifiers from rounds $0, 1, \ldots, R$. This **unfolded** (sparse) fingerprint lives in a $2^{32}$-element space. Count fingerprints (ECFC in Rogers & Hahn's naming) keep how many times each identifier occurred.

**Fold.** To get a fixed-length bit vector of length $m$ (2,048 in the project), set bit $\mathrm{id} \bmod m$ for every identifier.

**Naming: radius versus diameter.** ECFP*n* is named by the *diameter* $n$ of the largest environment, so ECFP4 corresponds to radius 2 (two bonds out from the centre in every direction, i.e. up to four bonds across). RDKit's `GetMorganGenerator(radius=2)` is therefore "ECFP4-like" — the same algorithm with slightly different invariants and hash function, so the bits are not identical to Pipeline Pilot's original ECFP4, but the behaviour is equivalent. ECFP6 = radius 3.

#### Worked example 4 — ECFP4 of aspirin by hand

Use the atom numbering of Section 2.7. A feature's identity is defined by its environment, so we can reason with *classes of identical environments* instead of hash values.

*Radius 0 (atom types).* Group atoms by their invariants (element, heavy degree, hydrogens, ring?):

| Class | Atoms | Description |
|---|---|---|
| a | 0 | CH₃, chain |
| b | 1, 10 | carbon with 3 heavy neighbours and no H, chain |
| c | 2, 11 | O with one (double) bond, no H |
| d | 3 | O with two heavy neighbours |
| e | 4, 9 | ring carbon, no H |
| f | 5, 6, 7, 8 | ring CH |
| g | 12 | O with one heavy neighbour and one H |

**7 distinct radius-0 features.**

*Radius 1 (atom + neighbours).* Write each as "centre {neighbour classes}":

| Atom(s) | Environment | Distinct? |
|---|---|---|
| 0 | a {b} | new |
| 1 | b {a, =c, d} | new |
| 2, 11 | c {=b} | same for both (both are C=O oxygens on a class-b carbon) |
| 3 | d {b, e} | new |
| 4 | e {d, f, e} | new |
| 5, 8 | f {e, f} | same |
| 6, 7 | f {f, f} | same |
| 9 | e {e, f, b} | new |
| 10 | b {e, =c, g} | new — differs from atom 1 because its neighbours differ |
| 12 | g {b} | new |

**10 distinct radius-1 features.** No duplicate bond sets yet (every radius-1 environment is a different "star" of bonds).

*Radius 2.* Each atom's environment now spans two bonds. Duplicate-structure removal bites here: atom 0's radius-2 environment covers bonds {0–1, 1=2, 1–3} — exactly atom 1's radius-1 environment — so it is dropped; likewise atom 2 (same bonds as atom 1 at radius 1), atom 11 and atom 12 (both equal to atom 10's radius-1 star). That leaves atoms 1, 3, 4, 5, 6, 7, 8, 9, 10. Atoms 6 and 7 still have identical environments (each sees the ring from a symmetric position), while atoms 5 and 8 now differ (atom 5's two-bond neighbourhood reaches the ester oxygen, atom 8's reaches the acid carbon). **8 distinct radius-2 features.**

Total: $7 + 10 + 8 = 25$ distinct identifiers. RDKit's unfolded Morgan fingerprint of aspirin has exactly 25 features, and Section 11 reproduces this count with a 30-line pure-Python re-implementation.

*Folding.* After reducing modulo 2,048, RDKit's bit vector has **24** bits, not 25: the radius-0 identifier of the hydroxyl oxygen (atom 12) and that of the carbonyl carbons (atoms 1 and 10) happen to land on the same bit (bit 807). That is a **bit collision**, in one of the simplest drugs in the data set.

#### How often do bits collide? (derivation)

Assume a molecule has $n$ distinct features and the hash spreads them uniformly and independently over $m$ bits. For a particular bit, the probability that *no* feature lands on it is $(1 - 1/m)^n$. By linearity of expectation, the expected number of bits that are set is
$$
\mathbb{E}[\#\text{set bits}] = m\left[1 - \left(1 - \tfrac{1}{m}\right)^{n}\right].
$$
Expanding $(1-1/m)^n = 1 - \frac{n}{m} + \binom{n}{2}\frac{1}{m^2} - \cdots$ gives
$$
\mathbb{E}[\#\text{set bits}] \approx n - \frac{n(n-1)}{2m},
\qquad
\mathbb{E}[\#\text{lost to collisions}] \approx \frac{n(n-1)}{2m}.
$$
This is the birthday problem: collisions grow with the *square* of the number of features. For aspirin ($n = 25$, $m = 2048$) we expect 0.15 lost bits — aspirin was simply unlucky. For donepezil ($n = 47$) we expect about 0.52 at 2,048 bits and 1.04 at 1,024 bits; for a 200-feature peptide at 1,024 bits, about 18. Across the 667 project drugs with a structure, about half (49.8%) lose at least one bit to an internal collision at 2,048 bits (mean 0.77 bits, maximum 6; Section 11.5).

Collisions matter more for *pairs*: a feature of molecule A and an unrelated feature of molecule B can land on the same bit and create a spurious "shared" bit. Folding therefore tends to **inflate** similarities and compress their range. Shorter fingerprints (1,024 bits, as in the CDK benchmark) inflate more.

### 6.5 FCFP: functional-class fingerprints

ECFP environments are chemically specific: an –OH and an –NH₂ are different atoms. Sometimes we care about *what an atom does* rather than *what it is*. **FCFP** (functional-class fingerprints) replace the initial invariants by **pharmacophoric roles**: hydrogen-bond donor, hydrogen-bond acceptor, aromatic, halogen, basic (positively ionisable), acidic (negatively ionisable). The iteration is identical. In RDKit you get FCFP4 by passing `atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()`.

Effects on project drugs (Section 11 prints these):

| Pair | ECFP4 | FCFP4 |
|---|---|---|
| donepezil / rivastigmine (two Alzheimer cholinesterase inhibitors, very different scaffolds) | 0.122 | 0.326 |
| imipramine / donepezil | 0.152 | 0.341 |
| imipramine / desipramine | 0.611 | 0.562 |

FCFP raises similarity between different scaffolds that present similar functional groups ("scaffold hopping"), but it also raises the background similarity of unrelated pairs, so it is not automatically better.

### 6.6 Options and blind spots of ECFP

* **Stereochemistry is ignored by default.** RDKit's Morgan generator uses `includeChirality=False`. Citalopram (racemic, no stereo marks) and escitalopram, and even the diastereomers quinine and quinidine, all get **Tanimoto 1.0**. With `includeChirality=True` the values drop to 0.958 and 0.787. The project uses the default, which is reasonable (stereoisomers usually share targets) but means stereo-driven differences in indication are invisible.
* **Double-bond geometry is ignored too.** The retinoids tretinoin (all-*trans* retinoic acid), alitretinoin (9-*cis*) and isotretinoin (13-*cis*) are *E/Z* isomers of one another and all three pairs get ECFP4 Tanimoto 1.0 in Fdataset.
* **Radius limits what can be seen.** Trihexyphenidyl (DB00376) and procyclidine (DB00387), two anticholinergic anti-Parkinson drugs, differ only in a piperidine (6-membered) versus pyrrolidine (5-membered) ring. Every radius-≤2 environment of one also occurs in the other (the cyclohexyl ring supplies the "CH₂ between two CH₂" environments the pyrrolidine lacks), so even the *unfolded* ECFP4 sets are identical: Tanimoto 1.0. Ring *size* is not an atom invariant.
* **Counts versus bits.** Bit vectors record presence only; a molecule with one benzene ring and one with four give the same benzene bits. Count fingerprints fix this.
* **Small molecules have few bits.** Amantadine has 13 bits; one feature more or less changes its Tanimoto with anything by several hundredths. Lithium (`[Li+]`, DB01356) has a single bit.

### 6.7 Other fingerprints, briefly

* **Atom pairs** (Carhart et al., 1985): features are (atom type, topological distance, atom type) triples — good at capturing global shape.
* **Topological torsions**: four-atom linear paths with atom types.
* **MinHashed fingerprints (MHFP, MAP4)**: circular features with a locality-sensitive hash, suited to very large libraries.
* **Learned fingerprints**: a graph neural network trained to embed molecules (the "neural fingerprint" of Duvenaud et al., 2015). They replace the fixed hash with learned message passing — the same idea as the GNN layers elsewhere in this course.

---

## 7. Similarity coefficients

### 7.1 Definitions

For two bit vectors $A$ and $B$ let
$a = |A|$ = number of bits set in $A$, $b = |B|$, and $c = |A \cap B|$ = number set in both. Then

| Coefficient | Formula | Range | Notes |
|---|---|---|---|
| **Tanimoto** (= Jaccard on sets) | $T = \dfrac{c}{a + b - c} = \dfrac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$ | [0, 1] | the standard in cheminformatics |
| **Dice** (Sørensen) | $D = \dfrac{2c}{a + b}$ | [0, 1] | weights shared bits twice |
| **Cosine** (Ochiai) | $C = \dfrac{c}{\sqrt{ab}}$ | [0, 1] | geometric mean of $c/a$ and $c/b$ |
| **Tversky** | $S_{\alpha,\beta} = \dfrac{c}{\alpha(a-c) + \beta(b-c) + c}$ | [0, 1] | asymmetric; $\alpha=\beta=1$ gives Tanimoto, $\alpha=\beta=\tfrac12$ gives Dice |

The **Soergel distance** $1 - T$ is a true metric (it satisfies the triangle inequality), which is one reason Tanimoto is popular: similarity-search data structures can rely on it.

For **count** vectors $x, y \in \mathbb{N}^m$ the natural generalisation ("MinMax") is
$$
T(x, y) = \frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)} ,
$$
which reduces to the set formula for 0/1 vectors. (Another variant, $\frac{x\cdot y}{\|x\|^2 + \|y\|^2 - x\cdot y}$, also reduces to it for binary vectors.)

### 7.2 Worked example 5 — four coefficients by hand

Take two 16-bit fingerprints:

```
position   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
A          1 1 0 1 0 1 1 0 1 0  0  0  0  0  1  1     a = 8
B          1 0 0 1 0 1 1 1 1 0  0  0  0  0  0  1     b = 7
A AND B    1 0 0 1 0 1 1 0 1 0  0  0  0  0  0  1     c = 6
A OR B     1 1 0 1 0 1 1 1 1 0  0  0  0  0  1  1     |A ∪ B| = 9 = a + b − c
```

* Tanimoto $T = 6/9 = 0.667$.
* Dice $D = 12/15 = 0.800$.
* Cosine $C = 6/\sqrt{56} = 0.802$.
* Tversky with $\alpha=1, \beta=0$: $6/(2 + 0 + 6) = 6/8 = 0.750$ — "what fraction of A's features are in B". With $\alpha=0, \beta=1$: $6/7 = 0.857$ — "what fraction of B's features are in A". Asymmetric similarity is useful when one molecule is a fragment or a scaffold of the other.

### 7.3 Dice and Tanimoto always rank neighbours identically (proof)

From $T = c/(a+b-c)$ we get $a + b = c/T + c = c(1+T)/T$. Substituting into Dice:
$$
D = \frac{2c}{a+b} = \frac{2c\,T}{c(1+T)} = \frac{2T}{1+T}.
$$
The function $f(T) = 2T/(1+T)$ has derivative $f'(T) = 2/(1+T)^2 > 0$, so it is strictly increasing. Hence for any query, sorting candidates by Dice or by Tanimoto gives *exactly the same order*, and a $k$-nearest-neighbour method (like the one used to measure standalone view signal in this project) gives identical neighbours. Check: imipramine/desipramine has $T = 0.611$, so $D = 1.222/1.611 = 0.759$, which RDKit confirms.

Cosine is *not* a monotone function of Tanimoto alone, because it also depends on the sizes $a, b$. Counter-example (query $a = 10$): candidate $B_1$ with $b=2, c=2$ has $T = 0.20$, $C = 0.447$; candidate $B_2$ with $b=10, c=4$ has $T = 0.25$, $C = 0.40$. Tanimoto prefers $B_2$; cosine prefers $B_1$. Cosine is kinder to small candidates that are "contained in" the query.

### 7.4 What is a "high" similarity?

There is **no universal threshold**: the distribution of similarity values depends on the fingerprint, its length and the chemical space. Some reference points:

* For random pairs of drug-like ChEMBL molecules, Greg Landrum's analysis (RDKit blog) puts the 95th and 99th percentiles of Morgan radius-2 bit-vector Tanimoto at about 0.19 and 0.23, of the RDKit path fingerprint (limited to 5-bond paths) at about 0.30 and 0.37, and of MACCS keys at about 0.59 and 0.67. A MACCS similarity of 0.6 is "noise"; a Morgan2 similarity of 0.6 is a very close analogue.
* In Fdataset itself, over all pairs of drugs with structures: mean ECFP Tanimoto 0.098 (95th / 99th percentiles 0.179 / 0.261) versus mean CDK Tanimoto 0.184 (0.344 / 0.469). The same pair of drugs means different things on the two scales.
* The often-quoted rule "Tanimoto ≥ 0.85 means similar activity" comes from Daylight fingerprints in the 1990s. Martin, Kofron & Traphagen (2002) tested it and found that a compound with Tanimoto ≥ 0.85 to an active compound had only about a 30% chance of being active itself. Similar structure raises the probability of similar activity; it does not guarantee it.

Practical rule: **compare similarities only within one fingerprint type**, and judge a value by where it sits in the background distribution (e.g. its percentile), not by an absolute cut-off. Methods that use the $k$ nearest neighbours (as in the project's view-signal test and the graph construction) are naturally robust to scale differences, because only the *ranking* matters.

### 7.5 The similarity principle and activity cliffs

The **similarity property principle** (Johnson & Maggiora, 1990) states that structurally similar molecules tend to have similar properties, including biological activity. It is the foundation of ligand-based virtual screening, of QSAR and of chemistry-based drug repositioning: similar drugs → similar targets → similar indications.

It fails at **activity cliffs**: pairs of very similar molecules with very different potency or effect. Maggiora (2006) pictured the "activity landscape" as terrain over chemical space: mostly rolling hills (smooth SAR) with occasional cliffs. Guha and Van Drie quantified it with the Structure–Activity Landscape Index,
$$
\mathrm{SALI}(i,j) = \frac{|A_i - A_j|}{1 - \mathrm{sim}(i,j)} ,
$$
which is large when activity differs a lot although similarity is close to 1. Cliffs are common (Stumpfe & Bajorath, 2012 surveyed them across many targets). In this project the most extreme "cliffs" are invisible by construction: quinine (antimalarial) and quinidine (antiarrhythmic) have ECFP similarity 1.0 because stereo is ignored, so the drug view will happily propagate malaria indications to quinidine and arrhythmia indications to quinine. Chemistry is one view among several precisely because of such cases.

---

## 8. Molecular descriptors and the rule of five

Fingerprints describe *which substructures* a molecule has. **Descriptors** are numbers summarising physicochemical *properties*. The most common:

| Descriptor | Meaning | How it is computed (RDKit) |
|---|---|---|
| **MW** | molecular weight (g/mol), including hydrogens | sum of average atomic masses (`Descriptors.MolWt`) |
| **logP** | log₁₀ of the octanol/water partition coefficient: >0 means the molecule prefers fat-like environments (lipophilic) | Wildman–Crippen atom-contribution model (`Crippen.MolLogP`); an estimate, not a measurement |
| **TPSA** | topological polar surface area (Å²): surface belonging to N and O atoms and their H's | Ertl's fragment sums (`rdMolDescriptors.CalcTPSA`); a proxy for membrane permeability |
| **HBD** | hydrogen-bond donors | Lipinski's version: number of N–H and O–H bonds counted per atom (`Lipinski.NHOHCount`) |
| **HBA** | hydrogen-bond acceptors | Lipinski's version: number of N and O atoms (`Lipinski.NOCount`) |
| **Rotatable bonds** | flexible single bonds outside rings | `rdMolDescriptors.CalcNumRotatableBonds` |

Note that RDKit also has *pharmacophore-based* HBD/HBA counts (`CalcNumHBD`, `CalcNumHBA`) that differ from Lipinski's simple N/O counts: aspirin has 4 N+O atoms (Lipinski HBA = 4) but only 3 acceptors by RDKit's SMARTS-based definition, which does not count the O–H oxygen of the carboxylic acid (an –OH attached to a C=O). Always say which definition you use.

**Lipinski's rule of five** (Lipinski et al., 1997) came from analysing compounds that reached phase II clinical trials. Poor oral absorption or permeation is more likely when a compound has

* more than **5** H-bond donors,
* more than **10** H-bond acceptors,
* molecular weight above **500**,
* calculated logP above **5**.

(All thresholds are multiples of 5, hence the name.) Usually "more than one violation" is the warning sign. The authors themselves noted exceptions: substrates of biological transporters, and classes such as antibiotics, antifungals, vitamins and cardiac glycosides. Veber et al. (2002) added two more filters: ≤ 10 rotatable bonds and TPSA ≤ 140 Å².

Values for some project drugs (computed in Section 11):

| Drug | MW | logP | TPSA | HBD | HBA (N+O) | Violations |
|---|---|---|---|---|---|---|
| Aspirin | 180.16 | 1.31 | 63.6 | 1 | 4 | 0 |
| Imipramine | 280.42 | 3.88 | 6.5 | 0 | 2 | 0 |
| Donepezil | 379.50 | 4.36 | 38.8 | 0 | 4 | 0 |
| Amantadine | 151.25 | 1.91 | 26.0 | 2 | 1 | 0 |

Across the 667 project drugs with a structure (largest fragment, Lipinski's N/O definitions), 536 (80%) have no violation, 72 have one, 32 have two and 27 have three; the multiple violators are dominated by peptides, macrolides, glycosides and other large natural products (Exercise 9). For this project descriptors are *not* used as a similarity view, but they are essential vocabulary when reading the chemistry literature or when explaining case-study predictions ("this candidate is orally available and CNS-penetrant": low TPSA, moderate logP).

---

## 9. Biologics: why some drugs have no chemical similarity

A **biologic** is a drug made by or derived from living systems: peptides, proteins (hormones, enzymes), monoclonal antibodies, vaccines, complex polysaccharides such as heparin. The project's drugs without a structure:

| Fdataset drugs with no SMILES | Kind |
|---|---|
| Sermorelin, calcitonin (salmon), teriparatide | peptide hormones (29, 32 and 34 amino acids) |
| Glatiramer | random copolymer of four amino acids, variable length |
| Heparin, enoxaparin, ardeparin | heterogeneous sulfated polysaccharide mixtures |
| Conjugated estrogens | a mixture of many estrogen sulfates from natural sources |
| Porfimer sodium | an oligomeric mixture of porphyrin units |
| DB00510, DB01258, DB01402 | retired DrugBank IDs (no record left to look up) |
| Propoxyphene, folinic acid, dyphylline | ordinary small molecules whose PubChem lookup returned no CID at build time |

The last row is a useful honesty check: missing structure is not *always* "biologic". Those three could be recovered by a name-based PubChem lookup.

Why do biologics not fit the SMILES/fingerprint machinery?

1. **No single defined structure.** Heparin is a population of chains of different lengths and sulfation patterns; glatiramer is a random polymer; conjugated estrogens is a mixture. A SMILES describes one molecule.
2. **Size.** A monoclonal antibody (IgG) is about 150 kDa: roughly 1,300 amino acids and on the order of 10,000 heavy atoms. Its ECFP would set nearly every bit of a 2,048-bit vector, so it would look "similar" to every other protein and to many small molecules — the bits saturate.
3. **The wrong notion of similarity.** What matters for a protein drug is its sequence, fold and binding epitope. Two antibodies with 95% identical frameworks but different complementarity-determining regions bind completely different targets. Sequence alignment (BLAST, Smith–Waterman), structure comparison, or target-based similarity are the right tools.
4. **Modifications.** PEGylation, glycosylation, conjugated payloads (antibody–drug conjugates) are hard to express in SMILES at all.

Short peptides are an intermediate case: leuprolide and goserelin (9-residue GnRH analogues) *do* have SMILES in PubChem and get fingerprints: 87 and 91 heavy atoms, 110 and 118 ECFP4 bits, and a Tanimoto of 0.839 to each other (Section 11.11). Their similarity is real, but it is driven by the seven shared amino-acid residues — essentially a crude sequence comparison done through substructures.

**How the project copes.** `morgan_tanimoto` returns `NaN` for every pair involving a drug without a parsable structure, then sets the diagonal to 1. In the graph, such a drug simply has *no neighbours in the `chem_ecfp` view*. It still has the benchmark's `chem_cdk` view (the 2011 authors had DrugBank structures even for some peptides, so `chem_cdk` covers all 593 drugs), possibly the `gene_r` view, and its own known links. The model's view-level attention learns per node how much to trust each view. The project rule (from `HOW_IT_WORKS.md`): *we never invent a similarity value.* Imputing a fake value (say 0 or the mean) would tell the model "this drug is dissimilar to everything", which is false information, whereas "no edge" is honest.

---

## 10. CDK versus RDKit ECFP in this project, and why ECFP won

The two chemical views differ in four ways at once:

| | `chem_cdk` (benchmark) | `chem_ecfp` (ours) |
|---|---|---|
| Structures | DrugBank canonical SMILES, around 2011 | PubChem isomeric SMILES (DrugBank's deposited records), 2026 |
| Standardisation | none reported | largest fragment |
| Fingerprint | CDK hashed **path** fingerprint, default parameters (1,024 bits, path depth 8) | RDKit **Morgan radius 2** (ECFP4-like), 2,048 bits |
| Similarity | Tanimoto | Tanimoto |
| Coverage | all drugs (F 593/593) | drugs with a structure (F 578/593, C 656/663) |

Measured differences on Fdataset (Section 11 recomputes all of these):

* Background level: mean pairwise similarity 0.184 (CDK) vs 0.098 (ECFP); 95th percentile 0.344 vs 0.179.
* Agreement: Pearson correlation 0.61, Spearman 0.55 — related but far from interchangeable.
* Saturation: 33 distinct drug pairs have CDK similarity exactly 1.0, versus 15 for ECFP. The CDK "identical" pairs include imipramine/desipramine, amitriptyline/nortriptyline, amantadine/memantine, levonorgestrel/norethindrone, lovastatin/simvastatin and even medroxyprogesterone acetate/testosterone (ECFP 0.354). All 15 ECFP "identical" pairs are also identical under CDK, and every one is explainable. Thirteen are stereoisomer pairs: six enantiomer or racemate/enantiomer pairs (citalopram/escitalopram, omeprazole/esomeprazole, levofloxacin/ofloxacin, bupivacaine/levobupivacaine, amphetamine/dextroamphetamine, and hyoscyamine/"Npc209773" — the latter is PubChem's odd title for DB00572, atropine, the racemate of hyoscyamine), four diastereomer pairs (quinine/quinidine, ephedrine/pseudoephedrine, epirubicin/doxorubicin, betamethasone/dexamethasone) and the three *E/Z* pairs among the retinoic acids. One is a salt/parent pair (theophylline/aminophylline) and one is the trihexyphenidyl/procyclidine radius artefact.
* Standalone predictive signal (AUPR of a 10-nearest-neighbour scorer, warm start): 0.061 → 0.091 on F and 0.047 → 0.068 on C.

**Why ECFP outperforms — the mechanisms.**

1. **Resolution at the level that matters.** ECFP atom invariants include the hydrogen count, degree and ring membership, so methylation, ring fusion and substitution patterns change features. The CDK path labels are hydrogen-blind and branch-blind, so homologues collapse onto identical fingerprints (Section 6.3). When many drug pairs are "1.0", the nearest-neighbour list is dominated by ties and near-ties that carry little information about which drug actually shares indications.
2. **Less hashing noise.** Long paths (up to 8 bonds) generate many more features per molecule than radius-2 environments, and they are folded into half as many bits (1,024). By the collision formula of Section 6.4, the number of spurious shared bits grows roughly with $n^2/m$; the CDK view therefore has a higher, flatter background (mean 0.18 vs 0.10), which compresses the useful range of similarities.
3. **Local environments track pharmacology.** Binding to a protein pocket is determined by local functional groups and their immediate surroundings — exactly what circular environments encode. In the large open benchmark of Riniker & Landrum (2013), circular fingerprints (ECFP4, ECFP6) were among the best at ranking actives early, topological torsions were near the top under every metric, and MACCS keys ranked last together with a trivial atom-count baseline. They also found that, apart from those baselines, differences between fingerprints were smaller than differences between targets — a useful reminder not to over-interpret one benchmark.
4. **Cleaner input.** Largest-fragment standardisation removes counter-ion bits; current PubChem structures fix some errors in old records.

Two caveats keep this honest. First, the absolute AUPRs are low (chemical similarity alone is a weak predictor of *indications*, which depend on targets, disease biology and clinical history), so ECFP is "less weak", not "strong". Second, CDK still has one advantage — complete coverage, including peptides and heparins — so the project keeps **both** views and lets attention decide.

---

## 11. Hands-on code (RDKit, run in order)

Start Python (or Jupyter) **in the project root** with the project's interpreter:

```
"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe"
```

Every block below was executed in one session, in this order; the text under each block is the exact output. RDKit sometimes prints warnings (for example "Omitted undefined stereo" when an InChI is generated for a molecule with an unspecified stereocentre); they go to the error stream and are silenced here with `RDLogger.DisableLog`.

### 11.1 From SMILES to a molecular graph

```python
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")          # silence RDKit warnings

aspirin = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")   # PubChem's Kekulé SMILES
print("heavy atoms:", aspirin.GetNumAtoms(), " bonds:", aspirin.GetNumBonds(),
      " formula:", rdMolDescriptors.CalcMolFormula(aspirin))
print("idx sym arom deg(heavy) H ring")
for a in aspirin.GetAtoms():
    print(f"{a.GetIdx():3d} {a.GetSymbol():>3} {str(a.GetIsAromatic()):>5} "
          f"{a.GetDegree():6d} {a.GetTotalNumHs():4d} {str(a.IsInRing()):>5}")
for b in list(aspirin.GetBonds())[:5]:
    print("bond", b.GetBeginAtomIdx(), "-", b.GetEndAtomIdx(), b.GetBondType())

# cyclomatic number = |E| - |V| + components
amantadine = Chem.MolFromSmiles("C1C2CC3CC1CC(C2)(C3)N")
ncomp = len(Chem.GetMolFrags(amantadine))
print("amantadine: cyclomatic number", amantadine.GetNumBonds() - amantadine.GetNumAtoms() + ncomp,
      "| RDKit ring count", amantadine.GetRingInfo().NumRings(),
      "| formula", rdMolDescriptors.CalcMolFormula(amantadine))

# invalid SMILES give None, not an exception
for bad in ["C(C)(C)(C)(C)C", "c1cccc1", "c1ccnc1", "c1cc[nH]c1"]:
    m = Chem.MolFromSmiles(bad)
    print(f"{bad:16s} ->", Chem.MolToSmiles(m) if m is not None else None)
```

```text
heavy atoms: 13  bonds: 13  formula: C9H8O4
idx sym arom deg(heavy) H ring
  0   C False      1    3 False
  1   C False      3    0 False
  2   O False      1    0 False
  3   O False      2    0 False
  4   C  True      3    0  True
  5   C  True      2    1  True
  6   C  True      2    1  True
  7   C  True      2    1  True
  8   C  True      2    1  True
  9   C  True      3    0  True
 10   C False      3    0 False
 11   O False      1    0 False
 12   O False      1    1 False
bond 0 - 1 SINGLE
bond 1 - 2 DOUBLE
bond 1 - 3 SINGLE
bond 3 - 4 SINGLE
bond 4 - 5 AROMATIC
amantadine: cyclomatic number 3 | RDKit ring count 4 | formula C10H17N
C(C)(C)(C)(C)C   -> None
c1cccc1          -> None
c1ccnc1          -> None
c1cc[nH]c1       -> c1cc[nH]c1
```

### 11.2 Canonical SMILES, stereo and InChIKeys

```python
from rdkit.Chem import rdCIPLabeler
drugs = {
    "aspirin":      "CC(=O)OC1=CC=CC=C1C(=O)O",
    "imipramine":   "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",
    "rivastigmine": "CCN(C)C(=O)OC1=CC=CC(=C1)[C@H](C)N(C)C",
    "citalopram":   "CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
    "escitalopram": "CN(C)CCC[C@@]1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
    "L-alanine":    "N[C@@H](C)C(=O)O",
}
for name, smi in drugs.items():
    m = Chem.MolFromSmiles(smi)
    centres = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    print(f"{name:13s} canonical={Chem.MolToSmiles(m)}")
    print(f"{'':13s} no-stereo={Chem.MolToSmiles(m, isomericSmiles=False)}  centres={centres}")
    print(f"{'':13s} InChIKey={Chem.MolToInchiKey(m)}")

print(Chem.MolToInchi(Chem.MolFromSmiles(drugs["aspirin"])))
print(Chem.MolToInchi(Chem.MolFromSmiles(drugs["rivastigmine"])))

# the same molecule written three ways -> one canonical SMILES
print({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ["OCC", "C(O)C", "CCO"]})

# E/Z from / and \
for s in ["F/C=C/F", r"F/C=C\F"]:
    m = Chem.MolFromSmiles(s)
    print(s, [str(b.GetStereo()) for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2])

# salts change the InChIKey
for s in ["CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31", "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31.Cl"]:
    print(Chem.MolToInchiKey(Chem.MolFromSmiles(s)), s[-6:])
```

```text
aspirin       canonical=CC(=O)Oc1ccccc1C(=O)O
              no-stereo=CC(=O)Oc1ccccc1C(=O)O  centres=[]
              InChIKey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N
imipramine    canonical=CN(C)CCCN1c2ccccc2CCc2ccccc21
              no-stereo=CN(C)CCCN1c2ccccc2CCc2ccccc21  centres=[]
              InChIKey=BCGWQEUPMDMJNV-UHFFFAOYSA-N
rivastigmine  canonical=CCN(C)C(=O)Oc1cccc([C@H](C)N(C)C)c1
              no-stereo=CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1  centres=[(13, 'S')]
              InChIKey=XSVMFMHYUFZWBK-NSHDSACASA-N
citalopram    canonical=CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21
              no-stereo=CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21  centres=[(6, '?')]
              InChIKey=WSEQXVZVJXJVFP-UHFFFAOYSA-N
escitalopram  canonical=CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21
              no-stereo=CN(C)CCCC1(c2ccc(F)cc2)OCc2cc(C#N)ccc21  centres=[(6, 'S')]
              InChIKey=WSEQXVZVJXJVFP-FQEVSTJZSA-N
L-alanine     canonical=C[C@H](N)C(=O)O
              no-stereo=CC(N)C(=O)O  centres=[(1, 'S')]
              InChIKey=QNAYBMKLOCPYGJ-REOHCLBHSA-N
InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)
InChI=1S/C14H22N2O2/c1-6-16(5)14(17)18-13-9-7-8-12(10-13)11(2)15(3)4/h7-11H,6H2,1-5H3/t11-/m0/s1
{'CCO'}
F/C=C/F ['STEREOE']
F/C=C\F ['STEREOZ']
BCGWQEUPMDMJNV-UHFFFAOYSA-N CC=C31
XZZXIYZZBJDEEP-UHFFFAOYSA-N C31.Cl
```

### 11.3 Standardising the project's multi-fragment drugs

```python
import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize

drugs_all = pd.read_csv("data/interim/drugs_all.csv", dtype=str).fillna("")
print("drugs:", len(drugs_all), " with SMILES:", (drugs_all.smiles != "").sum())

largest = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()
multi = drugs_all[drugs_all.smiles.str.contains(".", regex=False)]
for r in multi.itertuples():
    mol = Chem.MolFromSmiles(r.smiles)
    parent = largest.choose(mol)
    neutral = uncharger.uncharge(parent)
    p = Chem.MolToSmiles(neutral)
    print(f"{r.drugbank_id} {r.name[:24]:24s} frags={len(Chem.GetMolFrags(mol))} -> "
          f"{p[:45] + ('...' if len(p) > 45 else '')}")
```

```text
drugs: 682  with SMILES: 667
DB00067 Pituitrin                frags=2 -> NC(=O)CCC1NC(=O)C(Cc2ccccc2)NC(=O)C(Cc2ccc(O)...
DB00115 Anacobin                 frags=3 -> C/C1=C2N=C(/C=C3N=C(/C(C)=C4\N[C@@](C)([C@@H]...
DB00200 Codroxomin               frags=3 -> C/C1=C2/N[C@H]([C@H](CC(N)=O)[C@@]2(C)CCC(=O)...
DB00325 Nitroferricyanide        frags=7 -> N=O
DB00364 Sucralfate (USAN:USP:INN frags=22 -> O=S(=O)([O][Al])OC[C@H]1O[C@@](COS(=O)(=O)[O]...
DB00464 Sodium Tetradecyl Sulfat frags=2 -> CCCCC(CC)CCC(CC(C)C)OS(=O)(=O)O
DB00515 trans-Diamminedichloropl frags=3 -> N
DB00653 Sulfuric acid magnesium  frags=2 -> O=S(=O)(O)O
DB00958 Carboplatin              frags=4 -> O=C(O)C1(C(=O)O)CCC1
DB00995 Auroafen                 frags=3 -> CC(=O)OC[C@H]1O[C@@H](S)[C@H](OC(C)=O)[C@@H](...
DB01223 Aminophylline            frags=3 -> Cn1c(=O)c2[nH]cnc2n(C)c1=O
DB01390 Sodium Bicarbonate       frags=2 -> O=C(O)O
```

### 11.4 ECFP from scratch, compared with RDKit

The function below is a faithful miniature of the Morgan/ECFP algorithm of Section 6.4: initial invariants, sorted-neighbour hashing, duplicate-environment removal by bond set, collection of all identifiers. It uses `zlib.crc32` as the hash, so the identifiers differ from RDKit's, but the *number of distinct features* must match RDKit's unfolded fingerprint.

```python
import zlib
from rdkit.Chem import rdFingerprintGenerator

def toy_ecfp(smiles, radius=2):
    """Return the list of distinct new identifiers per radius, and the total set."""
    m = Chem.MolFromSmiles(smiles)
    inv = lambda a: (a.GetAtomicNum(), a.GetTotalDegree(), a.GetTotalNumHs(),
                     a.GetFormalCharge(), a.GetIsotope(), int(a.IsInRing()))
    ids = {a.GetIdx(): zlib.crc32(repr(inv(a)).encode()) for a in m.GetAtoms()}
    env = {a.GetIdx(): frozenset() for a in m.GetAtoms()}      # bonds covered
    features = set(ids.values())
    per_radius = [len(set(ids.values()))]
    seen_envs = set()
    for r in range(1, radius + 1):
        new_ids, new_env = {}, {}
        for a in m.GetAtoms():
            i = a.GetIdx()
            nbrs = sorted((b.GetBondTypeAsDouble(), ids[b.GetOtherAtomIdx(i)])
                          for b in a.GetBonds())
            new_ids[i] = zlib.crc32(repr((r, ids[i], nbrs)).encode())
            covered = set(env[i])
            for b in a.GetBonds():
                covered.add(b.GetIdx())
                covered |= env[b.GetOtherAtomIdx(i)]
            new_env[i] = frozenset(covered)
        kept = set()
        for i in sorted(new_ids, key=lambda i: (len(new_env[i]), new_ids[i])):
            if new_env[i] in seen_envs:          # same substructure seen before
                continue
            seen_envs.add(new_env[i])
            kept.add(new_ids[i])
        per_radius.append(len(kept))
        features |= kept
        ids, env = new_ids, new_env
    return per_radius, features

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
examples = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "imipramine": "CN(C)CCCN1c2ccccc2CCc2ccccc21",
            "amantadine": "NC12CC3CC(CC(C3)C1)C2",
            "donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"}
print("molecule     toy per-radius  toy total  RDKit unfolded  RDKit 2048-bit")
for name, smi in examples.items():
    per, feats = toy_ecfp(smi)
    m = Chem.MolFromSmiles(smi)
    unfolded = len(gen.GetSparseCountFingerprint(m).GetNonzeroElements())
    folded = gen.GetFingerprint(m).GetNumOnBits()
    print(f"{name:12s} {str(per):14s} {len(feats):9d} {unfolded:15d} {folded:15d}")

# Where is aspirin's collision? bitInfo maps bit -> ((atom, radius), ...)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()
gen.GetFingerprint(Chem.MolFromSmiles(examples["aspirin"]), additionalOutput=ao)
for bit, envs in sorted(ao.GetBitInfoMap().items()):
    if len({r for _, r in envs}) > 1 or bit == 807:
        print("bit", bit, "<-", envs)
```

```text
molecule     toy per-radius  toy total  RDKit unfolded  RDKit 2048-bit
aspirin      [7, 10, 8]            25              25              24
imipramine   [7, 11, 11]           29              29              29
amantadine   [4, 5, 4]             13              13              13
donepezil    [9, 19, 19]           47              47              47
bit 807 <- ((1, 0), (10, 0), (12, 0))
```

Two things to notice in the first block. The cyclomatic number of amantadine is 3, but RDKit reports 4 rings: adamantane consists of four equivalent six-membered rings, any three of which are independent. RDKit returns a *symmetrized* smallest set of rings so that symmetric atoms get symmetric ring memberships, which can exceed the cyclomatic number. And the invalid SMILES really do come back as `None` — no exception is raised.

In the ECFP block, bit 807 is set both by the radius-0 environment of atoms 1 and 10 (the carbonyl carbons) and by the radius-0 environment of atom 12 (the hydroxyl oxygen): two different features, one bit.

### 11.5 Expected collisions versus reality

```python
import numpy as np

def expected_lost(n, m):
    """Expected number of features lost to collisions: n - E[set bits]."""
    return n - m * (1 - (1 - 1 / m) ** n)

for n, m in [(25, 2048), (47, 2048), (47, 1024), (200, 1024)]:
    print(f"n={n:3d} m={m:4d}  exact={expected_lost(n, m):.3f}  approx={n*(n-1)/(2*m):.3f}")

# reality on all project drugs with a structure
largest = rdMolStandardize.LargestFragmentChooser()
lost = []
for s in drugs_all.smiles[drugs_all.smiles != ""]:
    mol = largest.choose(Chem.MolFromSmiles(s))
    n = len(gen.GetSparseCountFingerprint(mol).GetNonzeroElements())
    lost.append(n - gen.GetFingerprint(mol).GetNumOnBits())
lost = np.array(lost)
print(f"{len(lost)} drugs: {np.mean(lost > 0):.1%} have >=1 collision, "
      f"mean lost bits {lost.mean():.2f}, max {lost.max()}")
```

```text
n= 25 m=2048  exact=0.146  approx=0.146
n= 47 m=2048  exact=0.524  approx=0.528
n= 47 m=1024  exact=1.040  approx=1.056
n=200 m=1024  exact=18.239  approx=19.434
667 drugs: 49.8% have >=1 collision, mean lost bits 0.77, max 6
```

### 11.6 Similarity coefficients by hand and with RDKit

```python
from rdkit import DataStructs

A = np.array([1,1,0,1,0,1,1,0,1,0,0,0,0,0,1,1], dtype=bool)
B = np.array([1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1], dtype=bool)
a, b, c = A.sum(), B.sum(), (A & B).sum()
tanimoto = c / (a + b - c)
dice = 2 * c / (a + b)
cosine = c / np.sqrt(a * b)
tversky = lambda al, be: c / (al * (a - c) + be * (b - c) + c)
print(f"a={a} b={b} c={c}  T={tanimoto:.3f} D={dice:.3f} cos={cosine:.3f} "
      f"Tv(1,0)={tversky(1,0):.3f} Tv(0,1)={tversky(0,1):.3f}  2T/(1+T)={2*tanimoto/(1+tanimoto):.3f}")

# the same with RDKit bit vectors
fa, fb = DataStructs.ExplicitBitVect(16), DataStructs.ExplicitBitVect(16)
for i in np.flatnonzero(A): fa.SetBit(int(i))
for i in np.flatnonzero(B): fb.SetBit(int(i))
print(round(DataStructs.TanimotoSimilarity(fa, fb), 3), round(DataStructs.DiceSimilarity(fa, fb), 3),
      round(DataStructs.CosineSimilarity(fa, fb), 3), round(DataStructs.TverskySimilarity(fa, fb, 1, 0), 3))
```

```text
a=8 b=7 c=6  T=0.667 D=0.800 cos=0.802 Tv(1,0)=0.750 Tv(0,1)=0.857  2T/(1+T)=0.800
0.667 0.8 0.802 0.75
```

### 11.7 Comparing fingerprint families on project drugs

```python
from rdkit.Chem import MACCSkeys

smi = dict(drugs_all.set_index("name").smiles)      # name -> PubChem SMILES
names = ["Imipramine", "Desipramine", "Amitriptyline", "Chlorpromazine", "Donepezil",
         "Aspirin", "Amantadine", "Memantine", "Rivastigmine", "(-)-Galantamine",
         "Citalopram", "Escitalopram", "Quinine", "Quinidine", "Trihexyphenidyl", "Procyclidine"]
mols = {n: largest.choose(Chem.MolFromSmiles(smi[n])) for n in names}

ecfp  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ecfpc = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
fcfp  = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
rdkfp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
fps = {"ECFP4": lambda m: ecfp.GetFingerprint(m), "ECFP4*": lambda m: ecfpc.GetFingerprint(m),
       "FCFP4": lambda m: fcfp.GetFingerprint(m), "RDKit": lambda m: rdkfp.GetFingerprint(m),
       "MACCS": MACCSkeys.GenMACCSKeys}

pairs = [("Imipramine", "Desipramine"), ("Imipramine", "Amitriptyline"),
         ("Imipramine", "Chlorpromazine"), ("Imipramine", "Donepezil"), ("Imipramine", "Aspirin"),
         ("Amantadine", "Memantine"), ("Donepezil", "Rivastigmine"), ("Donepezil", "(-)-Galantamine"),
         ("Citalopram", "Escitalopram"), ("Quinine", "Quinidine"), ("Trihexyphenidyl", "Procyclidine")]
print(f"{'pair':32s}" + "".join(f"{k:>8s}" for k in fps))
for x, y in pairs:
    row = [DataStructs.TanimotoSimilarity(f(mols[x]), f(mols[y])) for f in fps.values()]
    print(f"{x[:15] + '/' + y[:15]:32s}" + "".join(f"{v:8.3f}" for v in row))
print("(ECFP4* = ECFP4 with includeChirality=True)")
print("on-bits ECFP4:", {n: ecfp.GetFingerprint(mols[n]).GetNumOnBits()
                         for n in ["Aspirin", "Amantadine", "Imipramine", "Donepezil"]})
```

```text
pair                               ECFP4  ECFP4*   FCFP4   RDKit   MACCS
Imipramine/Desipramine             0.611   0.611   0.562   0.987   0.854
Imipramine/Amitriptyline           0.415   0.415   0.469   0.240   0.675
Imipramine/Chlorpromazine          0.489   0.489   0.543   0.364   0.745
Imipramine/Donepezil               0.152   0.152   0.341   0.242   0.538
Imipramine/Aspirin                 0.128   0.128   0.175   0.162   0.094
Amantadine/Memantine               0.455   0.455   0.562   0.825   0.810
Donepezil/Rivastigmine             0.122   0.122   0.326   0.335   0.444
Donepezil/(-)-Galantamine          0.173   0.175   0.358   0.411   0.679
Citalopram/Escitalopram            1.000   0.958   1.000   1.000   1.000
Quinine/Quinidine                  1.000   0.787   1.000   1.000   1.000
Trihexyphenidyl/Procyclidine       1.000   1.000   1.000   0.989   0.917
(ECFP4* = ECFP4 with includeChirality=True)
on-bits ECFP4: {'Aspirin': 24, 'Amantadine': 13, 'Imipramine': 29, 'Donepezil': 47}
```

### 11.8 Why the CDK path fingerprint says imipramine = desipramine

We enumerate every linear path of up to 7 bonds, write it as a label string of elements and bond symbols (hydrogens are not part of the label, as in Daylight/CDK path fingerprints), read it in a canonical direction, and collect the *set* of labels. Any hashed path fingerprint is a function of this set.

```python
def path_label_set(smiles, max_bonds=7):
    m = Chem.MolFromSmiles(smiles)
    bond_sym = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}
    labels = {a.GetSymbol() for a in m.GetAtoms()}           # 0-bond paths
    for n_atoms in range(2, max_bonds + 2):                   # paths with 1..max_bonds bonds
        for path in Chem.FindAllPathsOfLengthN(m, n_atoms, useBonds=False):
            toks = [m.GetAtomWithIdx(path[0]).GetSymbol()]
            for i, j in zip(path, path[1:]):
                toks += [bond_sym[m.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()],
                         m.GetAtomWithIdx(j).GetSymbol()]
            fwd, rev = "".join(toks), "".join(reversed(toks))
            labels.add(min(fwd, rev))                         # direction-independent
    return labels

for x, y in [("Imipramine", "Desipramine"), ("Amantadine", "Memantine"),
             ("Imipramine", "Amitriptyline")]:
    Lx, Ly = path_label_set(smi[x]), path_label_set(smi[y])
    print(f"{x}/{y}: |Lx|={len(Lx)} |Ly|={len(Ly)} shared={len(Lx & Ly)} "
          f"Tanimoto={len(Lx & Ly) / len(Lx | Ly):.3f}")
```

```text
Imipramine/Desipramine: |Lx|=90 |Ly|=90 shared=90 Tanimoto=1.000
Amantadine/Memantine: |Lx|=16 |Ly|=16 shared=16 Tanimoto=1.000
Imipramine/Amitriptyline: |Lx|=90 |Ly|=93 shared=36 Tanimoto=0.245
```

### 11.9 Rebuilding the `chem_ecfp` view and comparing it with `chem_cdk`

This uses the project's own function on the Fdataset drug order and the benchmark's CDK matrix stored in `Fdataset.mat`.

```python
import sys, scipy.io as sio
sys.path.insert(0, "src")
from drepo.similarity import morgan_tanimoto
from scipy.stats import pearsonr, spearmanr

mat = sio.loadmat("data/raw/benchmarks/Fdataset.mat")
drug_ids = [str(x[0]) for x in mat["Wrname"].ravel()]
S_cdk = mat["drug"].astype(float)
tab = drugs_all.set_index("drugbank_id").loc[drug_ids]
S_ecfp = morgan_tanimoto(tab.smiles.tolist())
n_missing = int((np.isnan(S_ecfp).sum(1) == len(drug_ids) - 1).sum())   # row all-NaN except diagonal
print("shape", S_ecfp.shape, " drugs without ECFP:", n_missing)

iu = np.triu_indices(len(drug_ids), 1)
ok = ~np.isnan(S_ecfp[iu])
e, c = S_ecfp[iu][ok], S_cdk[iu][ok]
print(f"pairs={ok.sum()}  mean ECFP={e.mean():.3f} CDK={c.mean():.3f}")
print("95th/99th pct ECFP", np.percentile(e, [95, 99]).round(3), " CDK", np.percentile(c, [95, 99]).round(3))
print(f"Pearson={pearsonr(e, c)[0]:.3f}  Spearman={spearmanr(e, c)[0]:.3f}")
print("pairs == 1.0:  CDK", int((S_cdk[iu] > 0.9999).sum()), " ECFP", int((np.nan_to_num(S_ecfp[iu]) > 0.9999).sum()))

name = {d: (n if n else d) for d, n in zip(drug_ids, tab["name"])}   # fall back to the ID
print("ECFP == 1.0 pairs:")
for k in np.flatnonzero(np.nan_to_num(S_ecfp[iu]) > 0.9999):
    i, j = iu[0][k], iu[1][k]
    print(f"   {name[drug_ids[i]][:22]:22s} / {name[drug_ids[j]][:22]:22s} CDK={S_cdk[i, j]:.3f}")
```

```text
shape (593, 593)  drugs without ECFP: 15
pairs=166753  mean ECFP=0.098 CDK=0.184
95th/99th pct ECFP [0.179 0.261]  CDK [0.344 0.469]
Pearson=0.614  Spearman=0.553
pairs == 1.0:  CDK 33  ECFP 15
ECFP == 1.0 pairs:
   Amphetamine            / Dextroamphetamine      CDK=1.000
   Citalopram             / Escitalopram           CDK=1.000
   Theophylline           / Aminophylline          CDK=1.000
   Bupivacaine (USAN:INN: / Levobupivacaine        CDK=1.000
   Omeprazole             / Esomeprazole           CDK=1.000
   Trihexyphenidyl        / Procyclidine           CDK=1.000
   Hyoscyamine            / Npc209773              CDK=1.000
   Betamethasone          / Dexamethasone          CDK=1.000
   Epirubicin             / Doxorubicin            CDK=1.000
   Quinine                / Quinidine              CDK=1.000
   9-Cis-Retinoic Acid    / Retinoic Acid          CDK=1.000
   9-Cis-Retinoic Acid    / Isotretinoin           CDK=1.000
   Retinoic Acid          / Isotretinoin           CDK=1.000
   Pseudoephedrine        / Ephedrine              CDK=1.000
   Levofloxacin           / Ofloxacin              CDK=1.000
```

### 11.10 Descriptors and the rule of five

```python
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

def ro5(m):
    mw, logp = Descriptors.MolWt(m), Crippen.MolLogP(m)
    hbd, hba = Lipinski.NHOHCount(m), Lipinski.NOCount(m)
    viol = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
    return mw, logp, rdMolDescriptors.CalcTPSA(m), hbd, hba, int(viol)

print(f"{'drug':12s} {'MW':>7s} {'logP':>6s} {'TPSA':>6s} HBD HBA viol")
for n in ["Aspirin", "Imipramine", "Donepezil", "Amantadine"]:
    mw, lp, tpsa, hbd, hba, v = ro5(mols[n])
    print(f"{n:12s} {mw:7.2f} {lp:6.2f} {tpsa:6.1f} {hbd:3d} {hba:3d} {v:4d}")

rows = []
for r in drugs_all[drugs_all.smiles != ""].itertuples():
    rows.append((r.name, *ro5(largest.choose(Chem.MolFromSmiles(r.smiles)))))
ro = pd.DataFrame(rows, columns=["name", "MW", "logP", "TPSA", "HBD", "HBA", "viol"])
print(ro.viol.value_counts().sort_index().to_dict())
```

```text
drug              MW   logP   TPSA HBD HBA viol
Aspirin       180.16   1.31   63.6   1   4    0
Imipramine    280.42   3.88    6.5   0   2    0
Donepezil     379.50   4.36   38.8   0   4    0
Amantadine    151.25   1.91   26.0   2   1    0
{0: 536, 1: 72, 2: 32, 3: 27}
```

### 11.11 Peptides: fingerprints exist but mean something different

```python
pep = {n: largest.choose(Chem.MolFromSmiles(smi[n])) for n in ["Leuprolide", "Goserelin"]}
for n, m in pep.items():
    print(n, "heavy atoms:", m.GetNumAtoms(), " ECFP4 bits:", ecfp.GetFingerprint(m).GetNumOnBits())
print("Leuprolide/Goserelin ECFP4 Tanimoto:",
      round(DataStructs.TanimotoSimilarity(ecfp.GetFingerprint(pep["Leuprolide"]),
                                           ecfp.GetFingerprint(pep["Goserelin"])), 3))
missing = drugs_all[drugs_all.smiles == ""]
print(len(missing), "drugs without SMILES:", ", ".join(n if n else d for n, d in
                                                       zip(missing.name, missing.drugbank_id)))
```

```text
Leuprolide heavy atoms: 87  ECFP4 bits: 110
Goserelin heavy atoms: 91  ECFP4 bits: 118
Leuprolide/Goserelin ECFP4 Tanimoto: 0.839
15 drugs without SMILES: SERMORELIN, Calcitonin salmon, Conjugated estrogens, Ardeparin, DB00510, Propoxyphene, Folinic acid, Dyphylline, Porfimer Sodium, heparin, enoxaparin, DB01258, DB01402, Glatiramer, TERIPARATIDE
```

---

## 12. In this project: the code, line by line

### 12.1 Getting structures: `scripts/02_build_features.py::pubchem_drugs`

```python
# quoted from scripts/02_build_features.py, pubchem_drugs (abridged)
txt = _get(f"{PUG}/substance/sourceid/DrugBank/{db}/cids/TXT")
cid = txt.split()[0] if txt else ""
...
txt = _get(f"{PUG}/compound/cid/{chunk}/property/SMILES,InChIKey,Title/CSV")
```

* **Substance versus compound.** PubChem keeps two layers. A *substance* (SID) is a record exactly as a depositor (here DrugBank) submitted it. A *compound* (CID) is PubChem's standardised, de-duplicated structure. The first URL asks: "for the substance that DrugBank deposited under this DrugBank ID, which compound(s) does it map to?" — giving a CID without needing the DrugBank download. Biologics often have a substance but no compound; then the code falls back to the substance's first synonym for the name.
* `time.sleep(0.21)` keeps the script under PubChem's limit of 5 requests per second; `_get` retries on HTTP 503 (throttling).
* **Batch property fetch.** CIDs are sent 100 at a time to the property endpoint, requesting `SMILES` (the full isomeric SMILES, see Section 3.12), `InChIKey` and `Title`. These land in `data/interim/pubchem_drugs.csv` (a cache, so reruns do not hit the network) and in `drugs_all.csv`.
* Note what is *not* done here: no standardisation. PubChem compound records are already standardised by PubChem, but salts deposited as salts keep their counter-ions (Section 5.1). Standardisation happens at fingerprint time.

### 12.2 Fingerprints and similarity: `src/drepo/similarity.py::morgan_tanimoto`

```python
# quoted from src/drepo/similarity.py, morgan_tanimoto
def morgan_tanimoto(smiles: list[str | None], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    largest = rdMolStandardize.LargestFragmentChooser()

    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if mol is not None:
            mol = largest.choose(mol)   # drop counter-ions such as HCl
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)

    n = len(fps)
    S = np.full((n, n), np.nan)
    ok = [i for i, f in enumerate(fps) if f is not None]
    ok_fps = [fps[i] for i in ok]
    for a, i in enumerate(ok):
        sims = DataStructs.BulkTanimotoSimilarity(ok_fps[a], ok_fps)
        S[i, ok] = sims
    np.fill_diagonal(S, 1.0)
    return S
```

Line by line:

1. **Signature.** `radius=2` → ECFP4-like; `n_bits=2048` → folding length. The input is a list aligned with the benchmark's drug order, possibly containing empty strings or `None`.
2. **Imports inside the function** keep RDKit an optional dependency of the package (other functions in the module do not need it).
3. `RDLogger.DisableLog("rdApp.*")` silences RDKit's parse warnings, which would otherwise flood the log for odd records.
4. `GetMorganGenerator(radius, fpSize)` builds a reusable **fingerprint generator** (the modern RDKit API; the older `AllChem.GetMorganFingerprintAsBitVect` is deprecated). Defaults: no chirality, standard (ECFP-like) invariants, bit vector output.
5. `LargestFragmentChooser()` — the only standardisation step (Section 5.4), counting atoms including hydrogens.
6. The loop: `isinstance(s, str) and s` protects against `NaN` (pandas' float for missing) and empty strings; `MolFromSmiles` may still return `None` for an unparsable string, so the result is checked before `largest.choose`. Drugs without a molecule get `None` instead of a fingerprint.
7. `S = np.full((n, n), np.nan)` — **missing is NaN, never 0.** This is the "no invented similarity" rule in code.
8. `ok` lists the indices that have fingerprints; `BulkTanimotoSimilarity(query, list)` computes Tanimoto of one fingerprint against all others in C++, much faster than a Python double loop. Each call fills row `i` at the columns in `ok` (NumPy fancy indexing `S[i, ok]`).
9. `np.fill_diagonal(S, 1.0)` makes even missing drugs "similar to themselves". The caller treats a row whose only non-NaN entry is the diagonal as "uncovered": in `main()`, coverage is computed as `~np.isnan(S).all(1) & ~(np.nansum(S, 1) <= 1.0 + 1e-9)`.

Its call site:

```python
# quoted from scripts/02_build_features.py, main
dt = drug_tab.loc[drugs]                      # the benchmark's drug order
S_ecfp = sim.morgan_tanimoto(dt.smiles.tolist())
...
views_r = {"chem_cdk": S_cdk, "chem_ecfp": S_ecfp, "gene_r": S_gr}
```

`dt` is indexed by the benchmark's own DrugBank ID order, so row $i$ of `S_ecfp` is the same drug as row $i$ of the benchmark's `S_cdk` and of the association matrix `A`. Getting this alignment wrong is the most damaging silent bug possible in this pipeline (Section 13).

### 12.3 What happens downstream

The three drug views are stacked into `drug_views` in `data/processed/Fdataset.npz`. The graph builder turns each view into $k$-nearest-neighbour edges (NaN pairs never become edges), and MV-HGAT's view-level attention weights the views per node. A drug without a structure has no `chem_ecfp` edges and is represented through its other views — exactly the behaviour described in Section 9.

---

## 13. Common mistakes and misconceptions

1. **"Canonical SMILES identify a molecule across databases."** Only within one toolkit and version. Use InChIKeys for cross-database identity, or re-canonicalise everything with one toolkit.
2. **Fingerprinting the salt.** Forgetting to strip counter-ions inflates similarity among salts and deflates drug–salt similarity (theophylline vs aminophylline 0.839 instead of 1.0).
3. **Trusting "largest fragment" blindly.** It turns cisplatin into ammonia and strips platinum from carboplatin. Check records with metals.
4. **Mixing similarity scales.** "Similarity 0.6" means near-identical in ECFP4 and noise in MACCS. Compare values only within one fingerprint; use percentiles or ranks.
5. **Confusing radius and diameter.** ECFP4 = radius 2. `GetMorganGenerator(radius=4)` is ECFP8, not ECFP4.
6. **Assuming ECFP sees stereo.** It does not by default: enantiomers and diastereomers get 1.0.
7. **Treating a Tanimoto of 1.0 as "same molecule".** Different molecules can share all ECFP4 features (trihexyphenidyl/procyclidine) or all path labels (imipramine/desipramine in CDK).
8. **Reading too much into collisions — or too little.** One collision barely matters for a 2,048-bit ECFP4 of a drug; hundreds matter for 1,024-bit path fingerprints of large molecules.
9. **Filling missing similarities with 0.** Zero says "definitely dissimilar", which is false information. Keep NaN and let the model use other views.
10. **Using `RDKFingerprint` and calling it "Daylight".** It is Daylight-*like* (paths and branched subgraphs, different hashing); results differ from Daylight's.
11. **Comparing fingerprints from different toolkits** (CDK bits vs RDKit bits). Different aromaticity models and hash functions make the bits incomparable.
12. **Forgetting to check `MolFromSmiles` for `None`.** The next line then fails with a confusing `AttributeError` on `NoneType`.
13. **Misaligned rows.** Computing `S_ecfp` on `drugs_all.csv` order instead of the benchmark's order silently scrambles the view.
14. **Believing the rule of five is a law.** It is a heuristic for oral absorption of passively absorbed compounds; many successful drugs violate it.
15. **Thinking every missing SMILES is a biologic.** Three small molecules in this project are missing because of a failed lookup.

---

## 14. Exercises

Difficulty: ★ conceptual, ★★ calculation / short derivation, ★★★ coding or open-ended. Code solutions assume you have run Section 11 in the same session.

**Exercise 1 (★).** Decode `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O` (ibuprofen, DB01050): number the heavy atoms, list the bonds, find the implicit hydrogens and the molecular formula, and say how many rings it has and whether any atom is a stereocentre.

<details><summary>Solution</summary>

Walk: 0 C; 1 C (bond 0–1); branch `(C)` 2 C (1–2); 3 C (1–3); 4 `C1` (3–4, opens ring 1); 5 `=C` (4=5); 6 C (5–6); 7 `=C` (6=7); branch `(C=C1)`: 8 C (7–8), 9 `=C1` (8=9, closes ring: 9–4); then 10 C (7–10); branch `(C)` 11 C (10–11); 12 C (10–12); branch `(=O)` 13 O (12=13); 14 O (12–14).

So 15 heavy atoms, 15 bonds, one component: $\mu = 15 - 15 + 1 = 1$ ring (a benzene ring; atoms 4–9). Hydrogens: C0 3, C1 1, C2 3, C3 2, ring CH at 5, 6, 8, 9 (4 H), ring C4 and C7 none, C10 1, C11 3, C12 0, O13 0, O14 1 → 3+1+3+2+4+1+3+1 = 18. Formula C₁₃H₁₈O₂ (ibuprofen, MW 206.3).

Stereocentre: atom 10 has four different substituents (H, CH₃, COOH and the aryl ring), so it is chiral — but the SMILES has no `@`, so it describes the racemate (ibuprofen is sold as a racemic mixture; the *S* enantiomer is the active one). Atom 1 has two identical methyls, so it is not a stereocentre. Check:

```python
ibu = Chem.MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
print(rdMolDescriptors.CalcMolFormula(ibu), ibu.GetRingInfo().NumRings(),
      Chem.FindMolChiralCenters(ibu, includeUnassigned=True, useLegacyImplementation=False))
```

```text
C13H18O2 1 [(10, '?')]
```

`'?'` means "stereocentre with unspecified configuration".
</details>

**Exercise 2 (★).** Write SMILES for: (a) ethanol, (b) acetic acid, (c) benzene in Kekulé and aromatic form, (d) pyridine, (e) *cis*-2-butene, (f) ammonium chloride, (g) the (*R*) enantiomer of alanine. Verify with RDKit that (c) gives one canonical form.

<details><summary>Solution</summary>

(a) `CCO`; (b) `CC(=O)O`; (c) `C1=CC=CC=C1` and `c1ccccc1`; (d) `c1ccncc1`; (e) `C/C=C\C` (different slash symbols on the two sides → same side → *cis*/*Z*); (f) `[NH4+].[Cl-]`; (g) L-alanine is `N[C@@H](C)C(=O)O` (*S*), so swapping `@@` → `@` gives the mirror image `N[C@H](C)C(=O)O` (*R*, D-alanine).

```python
for s in ["CCO", "CC(=O)O", "C1=CC=CC=C1", "c1ccccc1", "c1ccncc1", r"C/C=C\C",
          "[NH4+].[Cl-]", "N[C@H](C)C(=O)O"]:
    m = Chem.MolFromSmiles(s)
    print(f"{s:16s} -> {Chem.MolToSmiles(m):16s}",
          Chem.FindMolChiralCenters(m, useLegacyImplementation=False) or "",
          [str(b.GetStereo()) for b in m.GetBonds() if str(b.GetStereo()) != "STEREONONE"] or "")
```

```text
CCO              -> CCO               
CC(=O)O          -> CC(=O)O           
C1=CC=CC=C1      -> c1ccccc1          
c1ccccc1         -> c1ccccc1          
c1ccncc1         -> c1ccncc1          
C/C=C\C          -> C/C=C\C           ['STEREOCIS']
[NH4+].[Cl-]     -> [Cl-].[NH4+]      
N[C@H](C)C(=O)O  -> C[C@@H](N)C(=O)O [(1, 'R')] 
```
</details>

**Exercise 3 (★★).** Two fingerprints have $a = 30$, $b = 20$ bits set and share $c = 15$. Compute Tanimoto, Dice, cosine, and Tversky with $(\alpha, \beta) = (0.9, 0.1)$. Verify that Dice $= 2T/(1+T)$.

<details><summary>Solution</summary>

$T = 15/(30 + 20 - 15) = 15/35 = 0.4286$. $D = 30/50 = 0.600$; check $2T/(1+T) = 0.8571/1.4286 = 0.600$ ✓. Cosine $= 15/\sqrt{600} = 15/24.495 = 0.612$. Tversky $= 15/(0.9 \cdot 15 + 0.1 \cdot 5 + 15) = 15/(13.5 + 0.5 + 15) = 15/29 = 0.517$. The Tversky weighting with $\alpha = 0.9$ penalises the 15 bits of A missing from B heavily and the 5 bits of B missing from A lightly, so it mostly measures "how much of A is in B".
</details>

**Exercise 4 (★★).** (a) Prove that $1 - T$ satisfies $0 \le 1-T \le 1$ and equals 0 iff $A = B$ (for non-empty sets). (b) Show with a counter-example that cosine and Tanimoto can rank two candidates differently. (c) Is the same true for Dice? Why?

<details><summary>Solution</summary>

(a) $0 \le c \le \min(a,b)$ and $a + b - c = |A \cup B| \ge c$, so $0 \le T \le 1$. $T = 1$ iff $|A \cap B| = |A \cup B|$ iff $A \cap B = A \cup B$ iff $A = B$. (The triangle inequality also holds — the Soergel/Jaccard distance is a metric — but its proof is longer.)

(b) Query $a = 10$. $B_1$: $b = 2$, $c = 2$ → $T = 2/10 = 0.20$, $\cos = 2/\sqrt{20} = 0.447$. $B_2$: $b = 10$, $c = 4$ → $T = 4/16 = 0.25$, $\cos = 4/10 = 0.40$. Tanimoto ranks $B_2$ first; cosine ranks $B_1$ first.

(c) No: $D = 2T/(1+T)$ is a strictly increasing function of $T$ alone (Section 7.3), so Dice and Tanimoto always produce the same ranking.
</details>

**Exercise 5 (★★).** Using the collision formula, (a) compute the expected number of lost bits for donepezil ($n = 47$) at $m = 256, 1024, 2048$; (b) what length $m$ keeps the expected loss below 0.1 for a molecule with $n = 60$ features? (c) Why do collisions tend to *increase* Tanimoto values between unrelated molecules?

<details><summary>Solution</summary>

(a) Using $n(n-1)/(2m) = 1081/m$: 4.22 at 256, 1.06 at 1024, 0.53 at 2048. The exact formula gives 3.99, 1.04 and 0.52 (the approximation overestimates when $n$ is not small compared with $m$). Compute exactly:

```python
for m in [256, 1024, 2048]:
    print(m, round(expected_lost(47, m), 3))
print("m for n=60, loss<0.1:", next(m for m in range(64, 100000) if expected_lost(60, m) < 0.1))
```

```text
256 3.986
1024 1.04
2048 0.524
m for n=60, loss<0.1: 17681
```

(b) $60 \cdot 59 / (2m) < 0.1 \Rightarrow m > 17{,}700$; the exact formula gives 17,681. In practice nobody uses such long vectors: collisions are accepted as noise.

(c) Similarity counts shared bits. A collision *within* a molecule merges two of its features into one bit (reducing $a$). A collision *between* molecules — feature $x$ of A and an unrelated feature $y$ of B landing on the same bit — creates a shared bit that does not correspond to any shared substructure, increasing $c$. Both effects push $T = c/(a+b-c)$ up. The shorter the vector and the more features per molecule, the higher the background similarity.
</details>

**Exercise 6 (★).** Quinine and quinidine get ECFP4 Tanimoto 1.0 in the project. (a) Why? (b) Name two ways to make the drug view distinguish them. (c) Would you, for this project? Argue both sides.

<details><summary>Solution</summary>

(a) They are diastereomers: same graph, different configuration at two stereocentres. Default Morgan invariants ignore chirality, so every environment is identical.

(b) Use `includeChirality=True` in the Morgan generator (Tanimoto drops to 0.787), or add a stereo-aware descriptor/fingerprint as an extra view. One could also use the full InChIKey to flag stereoisomer pairs.

(c) *For*: their clinical uses differ (antimalarial vs antiarrhythmic), so treating them as identical leaks indications between them — a false-positive source. *Against*: most stereoisomer pairs (citalopram/escitalopram, omeprazole/esomeprazole, betamethasone/dexamethasone) share indications; stereo-aware fingerprints would weaken those useful edges, and stereo annotation quality in public records is uneven (racemates often have no marks). A reasonable compromise is to keep the stereo-blind view and add the stereo-aware similarity as a separate view, letting attention decide.
</details>

**Exercise 7 (★★★).** Implement count-based Tanimoto (MinMax) for two `{feature: count}` dictionaries and check it against RDKit's Tanimoto on Morgan *count* fingerprints for imipramine vs chlorpromazine. Why can it differ from the bit-vector value?

<details><summary>Solution</summary>

```python
def minmax(x, y):
    keys = set(x) | set(y)
    return sum(min(x.get(k, 0), y.get(k, 0)) for k in keys) / \
           sum(max(x.get(k, 0), y.get(k, 0)) for k in keys)

cx = ecfp.GetSparseCountFingerprint(mols["Imipramine"])
cy = ecfp.GetSparseCountFingerprint(mols["Chlorpromazine"])
print("MinMax (ours):", round(minmax(cx.GetNonzeroElements(), cy.GetNonzeroElements()), 4))
print("RDKit counts :", round(DataStructs.TanimotoSimilarity(cx, cy), 4))
print("bit vectors  :", round(DataStructs.TanimotoSimilarity(ecfp.GetFingerprint(mols["Imipramine"]),
                                                             ecfp.GetFingerprint(mols["Chlorpromazine"])), 4))
```

```text
MinMax (ours): 0.5125
RDKit counts : 0.5125
bit vectors  : 0.4889
```

RDKit's Tanimoto on integer sparse vectors is $\sum \min / (\sum x + \sum y - \sum \min)$, which equals $\sum\min/\sum\max$ because $\min + \max = x + y$ element-wise, so the two numbers agree exactly. The count version differs from the bit version because repeated environments (several identical aromatic CH atoms, the two benzene rings) now contribute their multiplicities, and because the sparse fingerprint is unfolded (no collisions).
</details>

**Exercise 8 (★★★).** Write a function that flags drugs for which largest-fragment standardisation is chemically questionable: the original record contains a metal atom, and the chosen parent does not. Run it on `drugs_all.csv`.

<details><summary>Solution</summary>

```python
METALS = {"Li", "Na", "K", "Mg", "Ca", "Al", "Fe", "Pt", "Au", "Zn", "Cu", "Co", "Gd", "Bi", "Ag", "Ga", "Tc"}
def metal_lost(smiles):
    mol = Chem.MolFromSmiles(smiles)
    before = {a.GetSymbol() for a in mol.GetAtoms()} & METALS
    after = {a.GetSymbol() for a in largest.choose(mol).GetAtoms()} & METALS
    return sorted(before - after)

for r in drugs_all[drugs_all.smiles != ""].itertuples():
    lost = metal_lost(r.smiles)
    if lost:
        print(f"{r.drugbank_id} {r.name[:30]:30s} loses {lost} -> "
              f"{Chem.MolToSmiles(largest.choose(Chem.MolFromSmiles(r.smiles)))[:40]}")
```

```text
DB00115 Anacobin                       loses ['Co'] -> C/C1=C2N=C(/C=C3N=C(/C(C)=C4\[N-][C@@](C
DB00200 Codroxomin                     loses ['Co'] -> C/C1=C2/[N-][C@H]([C@H](CC(N)=O)[C@@]2(C
DB00325 Nitroferricyanide              loses ['Fe'] -> [N-]=O
DB00464 Sodium Tetradecyl Sulfate      loses ['Na'] -> CCCCC(CC)CCC(CC(C)C)OS(=O)(=O)[O-]
DB00515 trans-Diamminedichloroplatinum loses ['Pt'] -> N
DB00653 Sulfuric acid magnesium salt ( loses ['Mg'] -> O=S(=O)([O-])[O-]
DB00958 Carboplatin                    loses ['Pt'] -> O=C(O)C1(C(=O)O)CCC1
DB00995 Auroafen                       loses ['Au'] -> CC(=O)OC[C@H]1O[C@@H]([S-])[C@H](OC(C)=O
DB01390 Sodium Bicarbonate             loses ['Na'] -> O=C([O-])O
```

Sodium and magnesium are ordinary counter-ions — losing them is correct (for magnesium sulfate one could argue the magnesium *is* the drug, but the fingerprint of sulfate is no worse than that of a lone Mg²⁺). Platinum (cisplatin, carboplatin), gold (auranofin), cobalt (the two vitamin B12 forms) and iron (sodium nitroprusside, whose parent becomes `[N-]=O`) are essential parts of the drug. Cisplatin's parent becomes ammonia `N`. Sucralfate is *not* flagged: its aluminium atoms are bonded to sulfate oxygens in the SMILES, so they stay in the largest fragment.
</details>

**Exercise 9 (★★★).** Using the `ro` table from Section 11.10, list the drugs with 3 rule-of-five violations. What do they have in common?

<details><summary>Solution</summary>

```python
v3 = ro[ro.viol >= 3].sort_values("MW")
print(len(v3), "drugs; MW range", round(v3.MW.min()), "-", round(v3.MW.max()))
print(", ".join(v3.name.str.slice(0, 22)))
```

```text
27 drugs; MW range 509 - 6179
Argatroban, Daunorubicin, Doxorubicin, Epirubicin, Xylan Sulfate, Candesartan Cilexetil, Acarbose, Saquinavir, Itraconazole, Visudyne, Digoxin, Pimecrolimus, Rifampicin, Everolimus, Octreotide, 42-[3-Hydroxy-2-(hydro, Desmopressin, Pituitrin, Lhrh, Leuprolide, Goserelin, Anacobin, Codroxomin, Icatibant, Bleomycin, Heparin Pentasaccharid, GlyTouCan:G35236GY
```

They are almost all large natural products, peptides or oligosaccharides (MW 509 to 6,179, many N/O atoms and N–H/O–H groups): peptide hormones and analogues (octreotide, desmopressin, LHRH, leuprolide, goserelin, icatibant, pituitrin), anthracycline glycosides (daunorubicin, doxorubicin, epirubicin), macrolide immunosuppressants (everolimus, pimecrolimus, temsirolimus), the cardiac glycoside digoxin, antibiotics (rifampicin, bleomycin), the antifungal itraconazole, vitamin B12 forms, oligosaccharides (acarbose, xylan sulfate, the heparin pentasaccharide), plus saquinavir, argatroban, verteporfin (Visudyne) and the prodrug candesartan cilexetil. Many are injected; several belong to exactly the exception classes Lipinski listed (antibiotics, antifungals, vitamins, cardiac glycosides) — digoxin and rifampicin are taken orally despite three violations. This illustrates that the rule of five is about *passive oral absorption*, not about being a drug.
</details>

**Exercise 10 (★).** A colleague proposes joining the benchmark's drugs to CTD by drug name. Give three concrete failure modes, explain why the InChIKey is better, and say when even the InChIKey fails.

<details><summary>Solution</summary>

Name joins fail on (1) synonyms and brand names ("acetylsalicylic acid" vs "aspirin"; PubChem titles like "(-)-Galantamine" or "Sucralfate (USAN:USP:INN:BAN:JAN)" in our own table), (2) salt names ("imipramine hydrochloride" vs "imipramine"), (3) spelling/case/locale variants ("heparin" vs "Heparin", "acetaminophen" vs "paracetamol"). The InChIKey is computed from the structure by a single open algorithm, so different databases that store the same structure produce the same key regardless of naming; it is fixed-length and free of special characters. It fails for salts unless both sides are standardised (the key covers all components), for stereo when one side stores the racemate and the other an enantiomer (use the first block), for tautomers not covered by the standard InChI, and for biologics and mixtures that have no single structure — there, a name or a curated cross-reference is the only option. That is why the project's CTD mapping uses CID first, InChIKey second and name last.
</details>

**Exercise 11 (★★★).** For donepezil, list the five nearest Fdataset drugs by `chem_ecfp` and by `chem_cdk`. Which list looks pharmacologically more sensible?

<details><summary>Solution</summary>

```python
i = drug_ids.index("DB00843")                     # donepezil
for label, S in [("ECFP", S_ecfp), ("CDK", S_cdk)]:
    row = np.nan_to_num(S[i].copy(), nan=-1.0)
    row[i] = -1
    top = np.argsort(-row)[:5]
    print(label, [(name[drug_ids[j]][:18], round(float(S[i, j]), 2)) for j in top])
```

```text
ECFP [('Tetrabenazine', 0.3), ('Moexipril', 0.3), ('Prazosin', 0.29), ('Fentanyl', 0.26), ('Astemizole', 0.26)]
CDK [('Tetrabenazine', 0.61), ('Dextromethorphan', 0.5), ('DB01258', 0.49), ('Levobunolol', 0.48), ('Anisindione', 0.48)]
```

Donepezil = a 5,6-dimethoxy-indanone joined through a CH₂ to an N-benzylpiperidine. Every ECFP neighbour shares one of these two local motifs: tetrabenazine (a dimethoxy-benzo-fused ring with a ketone), moexipril (a 6,7-dimethoxy-tetrahydroisoquinoline), prazosin (a 6,7-dimethoxyquinazoline), fentanyl (an N-phenethylpiperidine) and astemizole (a piperidine with a methoxyphenethyl group). The list is chemically coherent. The CDK list starts with the same tetrabenazine, then mixes dextromethorphan, a retired DrugBank ID with no name, levobunolol and anisindione — harder to rationalise — and with values (0.48–0.61) that look "high" only because the CDK background is high (Section 10).

The sobering part: neither list contains galantamine or rivastigmine, the drugs that actually share donepezil's indication (Alzheimer disease): their ECFP similarity to donepezil is only 0.17 and 0.12. Drugs acting on the same target can have unrelated scaffolds, which is exactly why chemical similarity alone gives low AUPR and why the project combines it with other views.
</details>

**Exercise 12 (★★, open).** Propose a drug–drug similarity for heparin, enoxaparin and ardeparin that the project could use instead of NaN. What data would you need and what are the risks?

<details><summary>Solution</summary>

Options, from simplest to richest: (1) **ATC-code similarity** — all three are B01AB (heparin group antithrombotics), so a hierarchical similarity on the WHO ATC classification (the same Wang-type method as in Unit D3, applied to the ATC tree) would link them strongly; data: DrugBank or WHO ATC tables. (2) **Target-based similarity** — Jaccard of target sets (all bind antithrombin III); data: DrugBank targets or ChEMBL. (3) **A representative structure** — e.g. the heparin pentasaccharide (fondaparinux-like, which *does* have a SMILES in PubChem: DB00569 in our table) as a proxy for all three. Risks: ATC and target similarities encode *known therapeutic use*, which can leak the answer (indications) into the features — an evaluation-leakage issue covered in Unit E1; a proxy structure misrepresents polydisperse mixtures. Whatever is used should be a separate, clearly labelled view, not silently merged into `chem_ecfp`.
</details>

**Exercise 13 (★★).** Show that the unfolded radius-$R$ Morgan fingerprint of a molecule with $N$ heavy atoms has at most $N(R+1)$ features, and explain the three reasons real counts are smaller (use aspirin: $N = 13$, bound 39, actual 25).

<details><summary>Solution</summary>

In each of the $R+1$ rounds every atom produces exactly one identifier, so at most $N(R+1)$ identifiers exist. Real counts are lower because (1) **symmetry/equivalence**: atoms with identical environments produce identical identifiers (aspirin radius 0: 13 atoms → 7 classes); (2) **duplicate-structure removal**: environments covering the same bond set as an earlier one are dropped (aspirin radius 2: atoms 0, 2, 11, 12 dropped); (3) **saturation**: once an environment covers the whole molecule, larger radii add nothing new (not reached for aspirin at radius 2, but relevant for tiny molecules like lithium or ammonia). For aspirin: 13 → 7, 13 → 10, 13 → 8 (9 environments after removal, two of which coincide), total 25 < 39.
</details>

---

## 15. Answers to the PREREQUISITES.md self-check questions (D2)

### Q1. Why do we keep only the largest fragment of a molecule before fingerprinting?

Because a drug *record* is not always a single *molecule*. PubChem and DrugBank store many drugs as salts, hydrates or multi-component preparations, written in SMILES as dot-separated fragments: imipramine hydrochloride, sodium tetradecyl sulfate (`…[O-].[Na+]`), aminophylline (two theophyllines + ethylenediamine). A fingerprint is computed over every atom in the molecule object, so without standardisation:

* the **counter-ion's features pollute the drug's fingerprint**. Two unrelated hydrochloride salts share "Cl" bits; two sodium salts share sodium bits; similarity among salts is inflated, adding spurious edges to the drug graph;
* the **same active ingredient in different salt forms looks different**: theophylline vs aminophylline Tanimoto 0.839 with raw SMILES, 1.0 after standardisation. In a guilt-by-association model this matters: aminophylline and theophylline share indications (asthma, COPD), and we *want* the view to say they are the same drug;
* **identifiers stop matching**: the InChIKey of the salt differs from that of the parent, so database joins fail.

The pharmacologically active part is, for the overwhelming majority of salts, the largest organic fragment, so "keep the largest fragment" is a simple, deterministic rule that fixes all three problems with one line of code (`rdMolStandardize.LargestFragmentChooser().choose(mol)`).

Its limits should be stated in the same breath. RDKit counts atoms *including hydrogens*, so for cisplatin (`N.N.Cl[Pt+2]Cl`) the "largest fragment" is ammonia, and for carboplatin the platinum is discarded. Mixtures (Pituitrin = vasopressin + oxytocin) lose a component. These are a handful of drugs in our data (Exercise 8), so the overall view is barely affected, but a careful pipeline would whitelist known counter-ions instead of keeping the largest fragment, and keep metal complexes whole. Neutralisation (`Uncharger`) is a natural next step but matters little for ECFP, which encodes charge only locally.

### Q2. Why is chemical similarity missing for some drugs, and how does the model cope?

**Why missing.** `chem_ecfp` needs a SMILES. 15 Fdataset drugs (2.5%) and 7 Cdataset drugs (1.1%) have none, for three reasons:

1. **Biologics and mixtures have no single structure**: peptide hormones (sermorelin, salmon calcitonin, teriparatide), a random polymer (glatiramer), polysaccharide mixtures (heparin, enoxaparin, ardeparin), natural-source mixtures (conjugated estrogens), oligomeric mixtures (porfimer sodium). PubChem has a *substance* record for them but no *compound*, so there is no CID and no SMILES. Even if a SMILES existed, ECFP would be meaningless for a 150 kDa antibody: thousands of features would saturate the bit vector, and the relevant similarity is sequence/epitope similarity, not substructure overlap.
2. **Retired identifiers**: DB00510, DB01258 and DB01402 no longer resolve.
3. **Lookup failures for ordinary small molecules**: propoxyphene, folinic acid and dyphylline returned no CID at build time, although they have well-defined structures; a name-based fallback would recover them.

**How the model copes.**

* *Honest missingness.* `morgan_tanimoto` fills every pair involving such a drug with `NaN`, not 0. When the view is turned into a $k$-NN graph, NaN pairs never become edges, so the drug simply has **no neighbours in `chem_ecfp`**. Imputing 0 would assert "dissimilar to everything" (false), and imputing the mean would invent edges.
* *Redundant views.* The drug still has `chem_cdk` (the benchmark's CDK matrix covers all drugs, including several biologics, because the 2011 authors had DrugBank structures for them), possibly `gene_r` (CTD gene interactions; heparin has 80 CTD genes in our table), and its own known drug–disease links, which give it neighbours through the bipartite structure of the heterogeneous graph.
* *Learned trust.* MV-HGAT's view-level attention computes, per node, how much weight each view's message gets. A node with no `chem_ecfp` neighbours contributes nothing through that view, and attention shifts weight to the views that do carry information.
* *Coverage is reported*, not hidden: `02_build_features.py` prints per-view coverage, and `HOW_IT_WORKS.md` lists the missing drugs and why.

---

## 16. Summary and cheat sheet

**Molecules as graphs.** Vertices = heavy atoms (element, charge, H count, isotope, aromatic, chirality); edges = bonds (1, 2, 3, aromatic). Rings = $|E| - |V| + c$.

**SMILES grammar.**

| Syntax | Meaning |
|---|---|
| `C N O S P F Cl Br I B` | organic-subset atoms; implicit H by valence |
| `c n o s p` | aromatic atoms |
| `[13CH3+]` | isotope, symbol, H count, charge — no implicit H inside brackets |
| `-` `=` `#` `:` | single, double, triple, aromatic bond |
| `( )` | branch |
| `1`…`9`, `%10` | ring-closure labels |
| `.` | disconnected fragments (salts, mixtures) |
| `@` / `@@` | tetrahedral: anticlockwise / clockwise viewed from the first neighbour |
| `/` `\` | double-bond geometry; `F/C=C/F` = *E*, `F/C=C\F` = *Z* |

**Identifiers.** Canonical SMILES: one per molecule *per toolkit*. Isomeric = with stereo/isotopes. InChI: layered, open, toolkit-independent. InChIKey: 14-10-1 characters; first block = skeleton; `UHFFFAOYSA` = no stereo; last letter = protonation. Standardise before keying.

**Standardisation.** Clean up → largest fragment (counts H; ties by MW then SMILES) → neutralise → (tautomer) → recompute identifiers. Watch metals and mixtures.

**Fingerprints.**

| Type | Feature | Example |
|---|---|---|
| Structural keys | predefined SMARTS questions | MACCS 166 |
| Path-based | hashed linear paths (≤ 7–8 bonds) | Daylight, RDKit, CDK |
| Circular | hashed atom environments up to radius $R$ | ECFP4 (R=2), ECFP6 (R=3), FCFP |

ECFP: invariants → $R$ rounds of sorted-neighbour hashing → drop duplicate bond sets → set of identifiers → fold mod $m$. Expected collisions ≈ $n(n-1)/2m$.

**Similarity.** $T = c/(a+b-c)$; $D = 2c/(a+b) = 2T/(1+T)$ (same ranking); $\cos = c/\sqrt{ab}$; Tversky $c/(\alpha(a-c)+\beta(b-c)+c)$. Random-pair 95th percentile: ECFP4 ≈ 0.19, MACCS ≈ 0.59. Compare within one fingerprint only. Similarity principle holds on average; activity cliffs break it.

**Descriptors.** MW, logP, TPSA, HBD (NH+OH), HBA (N+O), rotatable bonds. Ro5: MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10; >1 violation = absorption risk.

**Project facts.** SMILES coverage 97.5% (F), 98.9% (C). `chem_ecfp` = RDKit Morgan r=2, 2,048 bits, largest fragment, Tanimoto; missing = NaN. `chem_cdk` = CDK path, 1,024 bits, depth 8. Standalone warm AUPR 0.061 → 0.091 (F), 0.047 → 0.068 (C).

---

## 17. Further resources (all links checked in October 2026)

DOI links resolve to the publisher's page; some publishers (notably ACS) refuse automated link checkers but open normally in a browser. "Free" means readable without a subscription.

**Core, free**

* RDKit — *Getting Started with the RDKit in Python* (free): https://www.rdkit.org/docs/GettingStartedInPython.html — the official tutorial; read the sections on molecules, fingerprints and similarity.
* RDKit Book (free): https://www.rdkit.org/docs/RDKit_Book.html — exact definitions of RDKit's aromaticity model, Morgan and RDKit fingerprints, feature definitions.
* RDKit `rdFingerprintGenerator` API (free): https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html — every generator option used in this chapter.
* RDKit `rdMolStandardize` API (free): https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html — LargestFragmentChooser, Uncharger, Cleanup.
* Daylight *SMILES Theory Manual* (free): https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html — the classic, readable SMILES reference, including stereo rules.
* OpenSMILES specification (free): http://opensmiles.org/opensmiles.html — the precise, vendor-neutral grammar.
* InChI Trust technical FAQ (free): https://www.inchi-trust.org/technical-faq/ — layers, InChIKey structure, collision discussion.
* TeachOpenCADD talktorials (free notebooks): https://projects.volkamerlab.org/teachopencadd/ — especially T002 (ADME, rule of five) https://projects.volkamerlab.org/teachopencadd/talktorials/T002_compound_adme.html and T004 (compound similarity, fingerprints) https://projects.volkamerlab.org/teachopencadd/talktorials/T004_compound_similarity.html.
* Greg Landrum, "Thresholds for 'random' in fingerprints the RDKit supports" (free blog post): https://greglandrum.github.io/rdkit-blog/posts/2021-05-18-fingerprint-thresholds1.html — background similarity distributions per fingerprint.
* PubChem PUG-REST documentation (free): https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest — the API used by `pubchem_drugs`.

**Papers**

* Rogers & Hahn (2010), "Extended-Connectivity Fingerprints", *J. Chem. Inf. Model.* 50:742 (paid; often free via institutional access): https://doi.org/10.1021/ci100050t — the ECFP algorithm, invariants, duplicate removal.
* Weininger (1988), "SMILES, a chemical language and information system. 1." *J. Chem. Inf. Comput. Sci.* 28:31 (paid): https://doi.org/10.1021/ci00057a005 — the original SMILES paper.
* Heller et al. (2015), "InChI, the IUPAC International Chemical Identifier", *J. Cheminform.* 7:23 (free): https://doi.org/10.1186/s13321-015-0068-4.
* Pletnev et al. (2012), "InChIKey collision resistance: an experimental testing", *J. Cheminform.* 4:39 (free): https://doi.org/10.1186/1758-2946-4-39.
* Maggiora, Vogt, Stumpfe & Bajorath (2014), "Molecular Similarity in Medicinal Chemistry", *J. Med. Chem.* 57:3186 (paid): https://doi.org/10.1021/jm401411z — the best single review of similarity concepts.
* Bajusz, Rácz & Héberger (2015), "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?", *J. Cheminform.* 7:20 (free): https://doi.org/10.1186/s13321-015-0069-3.
* Martin, Kofron & Traphagen (2002), "Do structurally similar molecules have similar biological activity?", *J. Med. Chem.* 45:4350 (paid): https://doi.org/10.1021/jm020155c — the 0.85 rule tested.
* Stumpfe & Bajorath (2012), "Exploring activity cliffs in medicinal chemistry", *J. Med. Chem.* 55:2932 (paid): https://doi.org/10.1021/jm201706b.
* Riniker & Landrum (2013), "Open-source platform to benchmark fingerprints for ligand-based virtual screening", *J. Cheminform.* 5:26 (free): https://doi.org/10.1186/1758-2946-5-26.
* Durant et al. (2002), "Reoptimization of MDL keys for use in drug discovery", *J. Chem. Inf. Comput. Sci.* 42:1273 (paid): https://doi.org/10.1021/ci010132r — the MACCS keys.
* Lipinski et al. (1997, reprinted 2001), "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings", *Adv. Drug Deliv. Rev.* 23:3 (paid): https://doi.org/10.1016/S0169-409X(96)00423-1 — the rule of five.
* Willighagen et al. (2017), "The Chemistry Development Kit (CDK) v2.0", *J. Cheminform.* 9:33 (free): https://doi.org/10.1186/s13321-017-0220-4 — the toolkit behind `chem_cdk`.
* Gottlieb et al. (2011), "PREDICT: a method for inferring novel drug indications…", *Mol. Syst. Biol.* 7:496 (free): https://doi.org/10.1038/msb.2011.26 — origin of Fdataset and its CDK similarity.

**Book**

* Leach & Gillet, *An Introduction to Chemoinformatics*, revised edition, Springer 2007 (paid): https://link.springer.com/book/10.1007/978-1-4020-6291-9 — chapters 1 (representation), 3 (descriptors), 5 (similarity) are the textbook version of this unit.

---

## 18. Glossary

* **Activity cliff** — a pair of highly similar molecules with very different activity.
* **Aromaticity** — extra stability of planar, conjugated rings with $4n+2$ π-electrons; written with lower-case atoms in SMILES.
* **Biologic** — a drug produced by or derived from living systems (peptide, protein, antibody, polysaccharide); usually no single SMILES.
* **Bit collision** — two different features hashed to the same bit of a folded fingerprint.
* **Bracket atom** — a SMILES atom in `[...]`, used for non-organic-subset elements or non-default isotope, charge, H count or chirality.
* **Canonical SMILES** — the unique SMILES one toolkit produces for a molecule.
* **CDK** — Chemistry Development Kit, an open-source Java cheminformatics library; produced the benchmark's `chem_cdk` similarity.
* **CID / SID** — PubChem compound identifier (standardised structure) / substance identifier (a depositor's record).
* **Counter-ion** — the oppositely charged partner of a drug ion in a salt (e.g. chloride in a hydrochloride).
* **Cyclomatic number** — number of independent rings, $|E| - |V| + c$.
* **Descriptor** — a number describing a molecular property (MW, logP, TPSA…).
* **Diastereomers** — stereoisomers that are not mirror images (quinine/quinidine).
* **Dice coefficient** — $2c/(a+b)$; monotone in Tanimoto.
* **ECFP / Morgan fingerprint** — circular fingerprint of hashed atom environments; ECFP$n$ has diameter $n$ (ECFP4 = radius 2).
* **Enantiomers** — non-superimposable mirror-image stereoisomers.
* **FCFP** — functional-class version of ECFP using pharmacophoric atom roles.
* **Fingerprint** — fixed-length (usually binary) vector encoding the presence of structural features.
* **Folding** — reducing a large feature space to $m$ bits by taking identifiers modulo $m$.
* **Formal charge** — bookkeeping charge on an atom (e.g. +1 on ammonium N).
* **Heavy atom** — any non-hydrogen atom.
* **Implicit hydrogen** — a hydrogen not written in the SMILES but implied by the valence rule.
* **InChI / InChIKey** — IUPAC's layered, canonical identifier / its 27-character hashed form.
* **Isomeric SMILES** — SMILES including stereo and isotope information.
* **Kekulé structure / kekulisation** — a representation of an aromatic ring with explicit alternating single/double bonds / the act of finding one.
* **Largest-fragment rule** — standardisation that keeps the fragment with the most atoms (RDKit counts hydrogens).
* **Lipinski rule of five** — MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10; heuristic for oral absorption.
* **logP** — log₁₀ octanol/water partition coefficient; lipophilicity.
* **MACCS keys** — 166 predefined structural-key bits.
* **Neutralisation (uncharging)** — adding/removing protons to make charged groups neutral.
* **Path-based fingerprint** — hashed linear paths (and possibly subgraphs) of the molecular graph.
* **Salt** — an ionic compound of a drug ion and a counter-ion.
* **Similarity property principle** — similar structures tend to have similar properties.
* **SMILES** — a line notation writing the molecular graph as a depth-first walk.
* **Standardisation** — mapping records to one agreed parent representation.
* **Stereoisomers** — same graph, different 3-D arrangement.
* **Structural keys** — fingerprints with one predefined question per bit.
* **Tanimoto (Jaccard) coefficient** — $c/(a+b-c)$, shared over union.
* **Tautomers** — structures differing only in the position of a hydrogen (and double bonds).
* **TPSA** — topological polar surface area, sum of N/O fragment contributions.
* **Tversky index** — asymmetric similarity with weights $\alpha, \beta$ on the unshared bits.
* **Valence** — total bond order an atom normally forms (C 4, N 3, O 2, halogens 1).
