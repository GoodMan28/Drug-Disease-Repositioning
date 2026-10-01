# Unit D1 — Pharmacology and Drug Repositioning

*What a drug is, how it works, how it becomes a medicine, and why a computer's suggestion is only the first step on a very long road.*

---

## 0. About this chapter

**Prerequisites.**
- No biology or chemistry background is assumed: every term is defined when it first appears.
- **A3 (Probability and statistics)** for the attrition arithmetic in Section 3.8.
- **B2 (Evaluation and imbalanced data)** is helpful for Sections 3.14–3.15 and the "In this project" section, which refer to AUC, AUPR, precision@k and the "unknown = negative" assumption.
- A1 (Python / NumPy / pandas) to run the code. One code block uses RDKit (Unit D2 covers cheminformatics in depth; here you only need to know that a fingerprint turns a molecule into bits and Tanimoto measures overlap).

**Estimated study time.** 10–14 hours: about 6 h for the theory and case histories (Section 3), 2 h for the code (Section 4), 1 h for the project walk-through (Section 5), and 2–4 h for the exercises.

**Learning objectives.** After this unit you will be able to:

1. Define *drug*, *target*, *mechanism of action*, *indication*, *off-label use* and *polypharmacology*, and give an example of each from this project's data.
2. Distinguish small molecules from biologics (size, manufacture, route, regulation) and explain why biologics are missing from the project's chemical similarity views.
3. Name the main classes of drug targets and modes of action (agonist, antagonist, enzyme inhibitor, channel blocker, …).
4. Explain pharmacodynamics (what the drug does to the body) versus pharmacokinetics (what the body does to the drug, ADME), and compute half-life, steady-state concentration and accumulation from clearance and volume of distribution.
5. Draw the drug-development pipeline from discovery to Phase IV, with typical sizes, durations and success rates, and compute the overall probability of success from phase-transition rates.
6. Define drug repositioning (repurposing), explain why it is usually faster and cheaper than *de novo* development, and state its non-scientific obstacles (patents, exclusivity, funding).
7. Tell the verified histories of sildenafil, thalidomide, minoxidil, aspirin, metformin in oncology, baricitinib for COVID-19, and several others, and classify each by how the new use was found.
8. Describe the main families of computational repositioning: signature-based, target/structure-based, network-based, machine learning on similarity matrices (this project), knowledge graphs, and real-world-data mining.
9. State the guilt-by-association principle precisely, demonstrate it on the project data, and give at least six distinct ways in which it fails.
10. Lay out the validation ladder that turns a top-10 prediction into an approved indication, and explain why a computational prediction is a hypothesis, not a recommendation.

---

## 1. Motivation: what would a pharmacologist ask about our predictions?

Run `scripts/06_case_study.py` on Cdataset and the model's top new suggestion for **Alzheimer disease** is **amantadine**, with a score of 0.90. The script reports "49 clinical trials" for amantadine in Alzheimer's disease. A machine-learning reader may stop there: high score, plenty of trials, case closed.

A pharmacologist would ask a series of different questions, and this chapter teaches you to ask them too:

- **Why did the model suggest it?** The occlusion analysis says the evidence came from the *chemical* views (`chem_ecfp`, `chem_cdk`). Amantadine (1-aminoadamantane) is the parent structure of **memantine** (3,5-dimethyl-1-aminoadamantane), an approved Alzheimer's drug that is a known link in the benchmark. In the project's CDK fingerprint view the two molecules have Tanimoto similarity **1.000**; in ECFP4, **0.455**. This is guilt-by-association in its purest form: "a close chemical relative of an Alzheimer's drug might also be one".
- **Is the mechanism plausible?** Memantine works by blocking NMDA-type glutamate receptors. Amantadine also blocks NMDA receptors weakly, and has dopaminergic effects; it was approved as an influenza A antiviral in 1966 and became a Parkinson's disease treatment after a patient's Parkinsonian symptoms improved while the patient took it to prevent flu, and worsened when it was stopped (Schwab et al., 1969). So there is a mechanistic thread.
- **Is the evidence real?** We checked the 49 "trials" one by one through the ClinicalTrials.gov API. **None of them lists amantadine as an intervention; 48 list memantine.** The registry's search expands drug names through the MeSH vocabulary, where memantine is filed under amantadine. The "evidence" was an artefact of the same chemical relationship that produced the prediction.
- **Would it reach the brain at the right concentration, at a tolerable dose, in elderly patients?** That is pharmacokinetics and safety, and no similarity matrix encodes it.
- **What would it take to test it?** A clinical-trial programme, ethics approval, funding — for an off-patent generic, from whom?

The project's benchmarks themselves are full of pharmacological history. In Fdataset, **thalidomide** is linked to leprosy and multiple myeloma — not to morning sickness, for which it was originally marketed. **Sildenafil** is linked to primary pulmonary hypertension. **Minoxidil** to hypertension and (through an odd disease mapping we will discuss) alopecia. **Metformin** to type 2 diabetes and polycystic ovaries. Every one of those is a repositioning story. To build, evaluate and *explain* a repositioning model, you need to know what those stories are and what they teach.

---

## 2. A map of the chapter

```
  Section 3.1-3.3  What a drug is; small molecules vs biologics; targets and mechanisms
  Section 3.4-3.5  Pharmacodynamics; pharmacokinetics (ADME) with worked numbers
  Section 3.6      Indication, label, off-label use, polypharmacology
  Section 3.7-3.9  How a medicine is made: pipeline, phases, attrition, cost, regulation
  Section 3.10     Repositioning: definition, why faster/cheaper, obstacles
  Section 3.11     Case histories (verified)
  Section 3.12     Computational repositioning approaches
  Section 3.13     Guilt by association: when it works, when it fails
  Section 3.14-3.15 From prediction to patient; why a prediction is a hypothesis
```

---

## 3. Core theory

### 3.1 What is a drug?

**Working definition.** A *drug* is a substance given to change the body's biology in order to diagnose, cure, mitigate, treat or prevent disease. (The US Federal Food, Drug, and Cosmetic Act defines drugs in essentially these terms, by their *intended use*; a substance becomes a "drug" in the legal sense partly because of what it is meant to do.)

Several words are often confused:

- **Active ingredient / active moiety**: the molecule responsible for the effect (e.g. *sildenafil*). Drugs are often sold as **salts** (sildenafil *citrate*) to improve stability or solubility; the salt partner is not the active part. This is why the project's cheminformatics step keeps only the largest fragment of a molecule (Unit D2).
- **Drug product**: the active ingredient plus excipients in a specific form, strength and route (a 50 mg tablet, a topical solution).
- **Generic name vs brand name**: *sildenafil* is the generic (non-proprietary) name; *Viagra* and *Revatio* are two brands of the same molecule for two different indications, at different doses (Section 3.11.1).
- **Prodrug**: an inactive or less active compound that the body converts into the active one. Minoxidil, for example, is converted to minoxidil *sulfate* by sulfotransferase enzymes; enzyme activity in hair follicles helps explain why topical minoxidil works for some people and not others.
- **Dose**: how much, how often, by which route. The same molecule at a different dose or route can be a different medicine (Section 3.13.2).

In databases: **DrugBank** gives each drug an ID like `DB00203` (sildenafil); PubChem gives compounds a CID (sildenafil: 135398744). The project's rows are DrugBank IDs (Unit D4 covers identifiers).

### 3.2 Small molecules versus biologics

| | Small-molecule drugs | Biologics |
|---|---|---|
| Typical size | usually < ~900 Da (aspirin 180 Da, sildenafil 474 Da) | proteins of ~5–150 kDa (insulin ≈ 5.8 kDa; a monoclonal antibody ≈ 150 kDa) |
| Made by | chemical synthesis | living cells (recombinant expression), purification |
| Typical route | oral tablets possible | usually injection/infusion (digested if swallowed) |
| Structure | exactly defined; has a SMILES string | large, heterogeneous (glycosylation); no meaningful SMILES |
| Reach | can enter cells, sometimes cross the blood–brain barrier | mostly act outside cells or on cell surfaces |
| Immunogenicity | rare | can trigger anti-drug antibodies |
| US application | New Drug Application (NDA) | Biologics License Application (BLA) |
| Copies after patent expiry | **generics** (identical molecule, shown bioequivalent) | **biosimilars** (highly similar, not identical) |

A popular rule of thumb for oral small molecules is **Lipinski's "rule of five"** (Lipinski et al. 1997): poor absorption is more likely when a molecule has more than 5 hydrogen-bond donors, more than 10 hydrogen-bond acceptors, molecular weight above 500, or calculated logP above 5. It is a heuristic, not a law.

**Project relevance.** The two benchmarks are almost entirely small molecules from DrugBank. The ~2 % of drugs without a chemical structure in our data are biologics or mixtures (e.g. heparins, conjugated estrogens) and three retired DrugBank IDs (`HOW_IT_WORKS.md`, Section 2.3). For them the `chem_cdk`/`chem_ecfp` views are empty, and the model must rely on gene and association information.

### 3.3 Targets, receptors, enzymes and mechanism of action

#### 3.3.1 Targets

Most drugs act by binding a specific biological molecule, its **target** — almost always a protein. Santos et al. (2017) catalogued the efficacy targets of 1,578 FDA-approved drugs and found they act through only 893 human and pathogen biomolecules, of which **667 are human proteins**. A few protein families dominate:

- **Receptors** — proteins that receive a chemical signal and transmit it:
  - *G-protein-coupled receptors (GPCRs)*: cell-surface receptors for hormones and neurotransmitters (adrenaline, dopamine, serotonin, opioids). Beta-blockers like propranolol act here.
  - *Ligand-gated ion channels*: receptors that are channels (the NMDA glutamate receptor blocked by memantine; GABA-A receptors targeted by benzodiazepines).
  - *Nuclear receptors*: intracellular receptors that act as transcription factors (estrogen receptor — target of tamoxifen and raloxifene; glucocorticoid receptor — target of dexamethasone).
  - *Enzyme-linked receptors*, especially receptor tyrosine kinases (KIT, PDGFR).
- **Enzymes** — proteins that catalyse reactions: cyclo-oxygenases (aspirin), phosphodiesterase-5 (sildenafil), HMG-CoA reductase (statins), kinases (imatinib, baricitinib), viral reverse transcriptase (zidovudine).
- **Ion channels and transporters** — proteins that move ions or molecules across membranes: ATP-sensitive potassium channels (opened by minoxidil), serotonin transporter (blocked by SSRI antidepressants).

#### 3.3.2 Modes of action

How a drug changes its target's activity:

- **Agonist**: binds a receptor and activates it (morphine at opioid receptors; leuprolide at GnRH receptors).
- **Antagonist**: binds without activating, blocking the natural signal (propranolol at beta-adrenergic receptors).
- **Partial agonist**: activates, but less than the natural ligand at full occupancy.
- **Inverse agonist**: reduces a receptor's spontaneous (constitutive) activity.
- **Allosteric modulator**: binds a site other than the natural ligand's and changes the receptor's response.
- **Enzyme inhibitor**: reversible (competitive or non-competitive) or irreversible. Aspirin is an *irreversible* inhibitor: it transfers an acetyl group onto cyclo-oxygenase, permanently disabling that enzyme molecule.
- **Channel blocker / opener**: e.g. memantine blocks the NMDA channel pore; minoxidil opens K<sub>ATP</sub> channels.
- **Molecular glue / degrader**: a newer concept that rewrote thalidomide's story. Thalidomide binds **cereblon**, part of an E3 ubiquitin ligase (Ito et al. 2010), and changes which proteins the ligase marks for destruction. Its anti-myeloma relatives (lenalidomide) cause degradation of the transcription factors IKZF1 and IKZF3 (Krönke et al. 2014; Lu et al. 2014), while degradation of SALL4 has been linked to thalidomide's birth defects (Donovan et al. 2018).

#### 3.3.3 Mechanism of action (MoA)

The **mechanism of action** is the chain from molecular event to clinical effect. Sildenafil, for example:

```
 sildenafil --| PDE5 enzyme         (inhibits the enzyme that breaks down cGMP)
                 => cGMP stays high in smooth-muscle cells
                 => smooth muscle relaxes => blood vessels widen
                 => in penile tissue: erection (with sexual stimulation)
                 => in lung arteries: lower pulmonary pressure
```

The same molecular event produces different clinical benefits in different tissues. This is the most common route to repositioning: **same target, different disease that depends on that target's pathway**.

Some important drugs have an incompletely understood mechanism. Metformin lowers blood glucose mainly by reducing glucose production in the liver; its proposed molecular mechanisms (effects on mitochondrial respiration and on the energy sensor AMPK, among others) are still debated decades after its introduction. Clinical usefulness does not wait for mechanistic certainty.

### 3.4 Pharmacodynamics: what the drug does to the body

**Pharmacodynamics (PD)** studies the relationship between drug concentration at the site of action and the size of the effect.

**Dose–response.** As concentration $C$ rises, the effect $E$ usually rises and then saturates. The **Hill (E<sub>max</sub>) model** describes this:

$$E(C) = \frac{E_{\max}\, C^{n}}{\text{EC}_{50}^{\,n} + C^{n}},$$

where $E_{\max}$ is the maximal effect (**efficacy**), $\text{EC}_{50}$ is the concentration giving half the maximal effect (**potency**: lower EC<sub>50</sub> = more potent), and $n$ (the Hill coefficient) sets the steepness. At $C = \text{EC}_{50}$, $E = E_{\max}/2$ for any $n$. At $C = 9\,\text{EC}_{50}$ with $n = 1$: $E = 9/10\, E_{\max}$.

**Potency is not efficacy.** Drug X may reach its effect at 1 nM but plateau at 50 % of the possible effect; drug Y may need 1 µM but reach 100 %. Y is less potent but more efficacious.

**Therapeutic index (TI).** In animal studies classically defined as $\text{TD}_{50}/\text{ED}_{50}$ (the dose toxic to 50 % divided by the dose effective in 50 %); in clinical practice, the *therapeutic window* is the concentration range that is effective but not toxic. A repurposed use that needs a higher dose than the original can fall outside the window.

**Selectivity and side effects.** No drug binds only one protein. **On-target effects** in the wrong tissue and **off-target effects** at other proteins produce side effects — and, historically, many repositioning discoveries *started* as side effects: sildenafil's erections, minoxidil's hair growth, amantadine's improvement of Parkinsonian symptoms. A side effect is an observation that the drug does something biologically meaningful in that tissue.

### 3.5 Pharmacokinetics: what the body does to the drug (ADME)

**Pharmacokinetics (PK)** describes the time course of drug concentration. The four processes are abbreviated **ADME**:

- **Absorption**: getting from the site of administration into the blood. The **bioavailability** $F$ is the fraction of a dose that reaches the systemic circulation unchanged (100 % for intravenous; often lower orally because of incomplete absorption and **first-pass metabolism** in the gut wall and liver).
- **Distribution**: spreading from blood into tissues. The **volume of distribution** $V_d$ is the apparent volume needed to contain the total amount in the body at the plasma concentration: $V_d = \text{amount in body}/C_{\text{plasma}}$. A $V_d$ much larger than body water (~42 L in an adult) means the drug concentrates in tissues. Special barriers matter: the **blood–brain barrier** keeps most molecules out of the brain — critical for any Alzheimer's or Parkinson's repositioning hypothesis.
- **Metabolism**: chemical transformation, mostly in the liver, especially by **cytochrome P450 (CYP)** enzymes. Metabolism can inactivate a drug, activate a prodrug, or produce toxic metabolites; one drug inhibiting another's CYP enzyme causes **drug–drug interactions**.
- **Excretion**: removal, mainly via kidneys (urine) or bile. Overall elimination efficiency is the **clearance** $CL$, the volume of plasma cleared of drug per unit time (L/h).

**The key equations (one-compartment, first-order elimination).**

$$k = \frac{CL}{V_d}, \qquad t_{1/2} = \frac{\ln 2}{k} = \frac{0.693\, V_d}{CL}, \qquad C(t) = \frac{F\cdot\text{Dose}}{V_d}\,e^{-kt}\ \ (\text{IV bolus: }F=1).$$

For repeated doses every $\tau$ hours:

$$\bar C_{ss} = \frac{F\cdot\text{Dose}}{CL\cdot\tau},\qquad R = \frac{1}{1-e^{-k\tau}}\ \ (\text{accumulation factor}),$$

and the concentration reaches $1 - 2^{-n}$ of steady state after $n$ half-lives — about 94 % after 4 and 97 % after 5. A **loading dose** $= C_\text{target}\cdot V_d / F$ reaches the target immediately.

#### Worked example 1 — a drug with $V_d = 50$ L and $CL = 5$ L/h

$k = 5/50 = 0.1$ h<sup>−1</sup>; $t_{1/2} = 0.693/0.1 = 6.93$ h. Giving 100 mg intravenously every 8 h: $\bar C_{ss} = 100/(5 \times 8) = 2.5$ mg/L; $R = 1/(1 - e^{-0.8}) = 1/(1 - 0.4493) = 1.816$; steady-state peak $= (100/50)\times 1.816 = 3.63$ mg/L; trough $= 3.63\,e^{-0.8} = 1.63$ mg/L. Steady state is essentially reached after 4–5 half-lives, i.e. about 28–35 hours. Section 4.6 verifies these numbers by simulation.

**Why PK matters for repositioning.**

1. The new indication may need **different exposure**. Sildenafil for erectile dysfunction is taken as a single dose when needed (typically 50 mg); for pulmonary arterial hypertension it is taken continuously, three times a day (20 mg) — the same molecule, a different dosing regimen, a different product (Revatio).
2. The drug must **reach the target tissue**. A prediction for a brain disease is weak if the drug does not cross the blood–brain barrier.
3. The new route may change everything: oral minoxidil lowers blood pressure; topical minoxidil, with low systemic exposure, grows hair with fewer cardiovascular effects.
4. The new population may differ (children, elderly, pregnant women, kidney impairment), changing clearance and safety.

None of this is in the project's similarity views. That is one reason a high score is only a hypothesis.

### 3.6 Indication, label, off-label use, polypharmacology

- **Indication**: a disease or condition for which a drug is approved by a regulator, as stated in its official **label** (prescribing information / summary of product characteristics). The label also lists dose, contraindications, warnings and side effects. Our association matrix $A$ is a matrix of (mostly) indications.
- **Contraindication**: a situation in which the drug should not be used (e.g. testosterone products are contraindicated in men with known or suspected prostate cancer, because androgens can stimulate its growth — relevant to Section 5.5).
- **Off-label use**: a prescriber using an approved drug for an unapproved indication, dose, population or route. In the US this is legal and common: the FDA states that, once a drug is approved, health-care providers generally may prescribe it for an unapproved use when they judge it medically appropriate, although the FDA has not determined that the drug is safe and effective for that use. Companies may not *promote* off-label uses. Radley et al. (2006) estimated that about **21 %** of office-based prescriptions in the US were off-label, and that **73 %** of those off-label uses had little or no scientific support. Off-label use is both a source of repositioning ideas (clinicians notice what works) and a reason for caution (much of it is unproven).
- **Polypharmacology**: a drug acting on multiple targets. It explains side effects, but also creates repositioning opportunities. Imatinib was designed against the BCR-ABL kinase of chronic myeloid leukaemia (FDA approval May 2001), but it also inhibits the KIT kinase, which drives gastrointestinal stromal tumours — and it was approved for GIST in February 2002. "One drug, several targets, several diseases" is the molecular basis of many repositioning successes.

### 3.7 The drug-development pipeline

A new medicine typically takes **10–15 years** from the start of a discovery programme to approval.

```
 DISCOVERY            PRECLINICAL           CLINICAL DEVELOPMENT                     REVIEW      POST-MARKET
 (2-5 y)              (1-2 y)               (6-7 y in total)                          (~1 y)      (ongoing)
+-----------+      +--------------+   +-----------+  +------------+  +-------------+  +--------+  +-----------+
| target ID |      | in vitro and |   |  PHASE I  |  |  PHASE II  |  |  PHASE III  |  |NDA/BLA |  | PHASE IV  |
| screening | ---> | animal tests | ->| 20-100    |->| up to      |->| 300-3,000   |->| review |->| several   |
| hit->lead |      | toxicology,  |   | people;   |  | several    |  | patients;   |  | label  |  | thousand; |
| optimise  |      | PK, GLP      |   | safety,   |  | hundred    |  | 1-4 years;  |  |        |  | rare side |
|           |      |      |       |   | dose;     |  | patients;  |  | efficacy &  |  |        |  | effects,  |
|           |      |      v       |   | months    |  | months-2 y;|  | adverse     |  |        |  | new uses  |
|           |      |  IND filed   |   | ~70% move |  | efficacy & |  | reactions;  |  |        |  |           |
|           |      |  (permission |   | on        |  | side       |  | ~25-30%     |  |        |  |           |
|           |      |  to test in  |   |           |  | effects;   |  | move on     |  |        |  |           |
|           |      |  humans)     |   |           |  | ~33% move  |  |             |  |        |  |           |
+-----------+      +--------------+   +-----------+  +------------+  +-------------+  +--------+  +-----------+
   thousands of compounds  ->  a handful of candidates  ->  ~1 in 7 to 1 in 17 entering Phase I is approved
```

(Phase sizes, durations and transition percentages: US FDA, *The Drug Development Process*, Step 3.)

**Stage by stage.**

1. **Discovery.** Identify a *target* believed to drive the disease (genetics, biology); screen libraries of compounds for *hits* that modulate it; optimise hits into *leads* with better potency, selectivity and drug-like properties.
2. **Preclinical research.** Laboratory and animal studies of pharmacology, PK and toxicology under Good Laboratory Practice (GLP). In the US, the sponsor then files an **Investigational New Drug (IND)** application to start human trials.
3. **Phase I.** Usually 20–100 healthy volunteers (or patients, e.g. in oncology). Goals: safety, tolerability, PK, dose range. *Is it safe enough to give to people, and at what dose?*
4. **Phase II.** Up to several hundred patients. Goals: preliminary efficacy ("proof of concept"), dose–response, side effects. This is where most programmes fail.
5. **Phase III.** Typically 300–3,000 patients, often multicentre, randomised, double-blind, controlled against placebo or standard of care. Goals: confirm efficacy and monitor adverse reactions with enough statistical power to support approval.
6. **Regulatory review.** The sponsor submits a **New Drug Application (NDA)** or **Biologics License Application (BLA)**; regulators review the evidence and the proposed label.
7. **Phase IV / post-marketing surveillance.** After approval: safety monitoring in thousands to millions of patients (rare adverse events), additional studies, and — importantly for us — the discovery of new uses.

**Randomised controlled trials (RCTs)** are the core of Phase II/III: patients are randomly assigned to the drug or a control, so that the groups differ only by chance in everything except the treatment; *blinding* prevents patients' and doctors' expectations from biasing outcomes. Observational evidence (e.g. "diabetics on metformin get less cancer") cannot fully remove confounding; RCTs can. Section 3.11.5 shows how much this matters.

### 3.8 Attrition, cost and time

**Attrition** means most candidates fail.

- With the FDA's approximate phase-transition rates (70 % → 33 % → 25–30 %), the probability that a drug entering Phase I gets through Phase III is $0.70 \times 0.33 \times 0.25 \approx 0.058$ to $0.70 \times 0.33 \times 0.30 \approx 0.069$, i.e. **roughly 1 in 14 to 1 in 17** (Section 4.6).
- Wong, Siah & Lo (2019), analysing 406,038 trial records for more than 21,000 compounds (2000–2015), estimated an overall probability of approval from Phase I of **13.8 %**, and only **3.4 % in oncology**.
- Why do they fail? Harrison (2016) analysed 174 Phase II/III failures with stated reasons in 2013–2015: **52 % lack of efficacy, 24 % safety**, 15 % strategic, 6 % commercial, 3 % operational.

**Cost.** Because failures must be paid for, the cost *per approved drug* is far higher than the cost of one programme.

- DiMasi, Grabowski & Hansen (2016), using confidential data from 10 companies on 106 drugs, estimated **US$2.56 billion** (2013 dollars) per approval, including the cost of failures and the cost of capital (out-of-pocket: US$1.4 billion).
- Wouters, McKee & Luyten (2020), using public data on 63 drugs, estimated a **median of US$985 million and a mean of US$1.34 billion** (2018 dollars).
- The estimates differ by method, sample and therapeutic area; "about US$1–2.6 billion and 10–15 years" is a fair summary.

**Why this matters for repositioning.** If you start from a drug that already passed Phase I safety, has known PK and manufacturing, and perhaps decades of post-marketing safety data, you skip much of the most expensive and failure-prone early work — but not the efficacy trials, where most failures happen (52 % of late failures are lack of efficacy). Repositioning removes *safety* risk far more than *efficacy* risk.

### 3.9 Regulation in brief

- **United States (FDA).** New molecular entities: NDA under section 505(b)(1) of the Food, Drug, and Cosmetic Act, or BLA for biologics. A **new indication** for an approved drug is usually added via a **supplemental NDA (sNDA)** by the label holder, or through a **505(b)(2) NDA**, which may rely partly on published literature or on FDA's previous findings for the approved drug (FDA guidance, *Applications Covered by Section 505(b)(2)*, 1999). Incentives: **3-year exclusivity** for a new condition of use supported by new clinical investigations essential to approval (Hatch–Waxman, 1984); **7-year orphan-drug exclusivity** for diseases affecting fewer than 200,000 people in the US (Orphan Drug Act, 1983, as amended in 1984). Emergencies: **Emergency Use Authorization (EUA)**, as used for baricitinib and, briefly, hydroxychloroquine in COVID-19.
- **History that shaped the system.** The thalidomide disaster (Section 3.11.2) led directly to the **Kefauver–Harris Drug Amendments**, signed on 10 October 1962, which required manufacturers to provide substantial evidence of *effectiveness* from adequate and well-controlled studies, not just safety.
- **Europe** has the European Medicines Agency (EMA) with analogous procedures; **India** regulates through the Central Drugs Standard Control Organisation (CDSCO) under the New Drugs and Clinical Trials Rules, 2019.

### 3.10 Drug repositioning (repurposing)

**Definition.** *Drug repositioning* (also *repurposing*, *reprofiling*, *re-tasking*) is finding new uses for existing drugs — approved, withdrawn, shelved or failed in development — outside the scope of their original indication (Ashburn & Thor 2004; Pushpakom et al. 2019). Three flavours:

1. **New indication for an approved drug** (sildenafil → pulmonary hypertension).
2. **Rescue of a shelved or failed compound** (zidovudine, a failed anticancer compound from the 1960s, became the first HIV drug in 1987; sildenafil failed for angina before succeeding for erectile dysfunction).
3. **New formulation, route or dose for a new use** (topical minoxidil; a paediatric oral propranolol solution for infantile haemangioma).

**Why it is faster and cheaper.** Nosengo (2016), reporting industry estimates, put repurposing at roughly **6.5 years and US$300 million** on average, versus roughly **13–15 years and US$2–3 billion** for a new drug; Pushpakom et al. (2019) cite the same figures. The savings come from:

- **Known safety in humans** (Phase I data, often years of post-marketing experience) → lower risk of safety failure; development can often start at Phase II.
- **Known PK, formulation and manufacturing** → no new chemistry programme.
- **Existing supply** → trials can start quickly (important in pandemics).
- **Lower regulatory burden** via 505(b)(2) or sNDA when prior data can be referenced.

**Why it is not easy (Begley et al. 2021; Pushpakom et al. 2019).**

- **Efficacy risk remains.** Most late-stage failures are for lack of efficacy, and repositioning does not remove that.
- **Dose may differ**, re-opening safety questions (Section 3.5).
- **Weak commercial incentive for generics.** If the drug is off patent, anyone can sell it; the company that pays for a Phase III trial may not recoup the cost, because doctors can prescribe cheaper generics off-label. Exclusivities (3-year, orphan) and method-of-use patents only partly help. Much repositioning of generics is therefore driven by academia, charities and governments (e.g. the US NIH's National Center for Advancing Translational Sciences, or the Anticancer Fund's ReDO project for oncology).
- **Data access.** Failed-trial data and shelved compounds sit in companies.

### 3.11 Case histories

Each history below was checked against primary or authoritative sources (listed in Section 10). Notice *how* each new use was found — that is what computational methods try to systematise.

#### 3.11.1 Sildenafil: angina → erectile dysfunction → pulmonary arterial hypertension

- **1986–1989.** Pfizer chemists at Sandwich, UK, searching for drugs for angina pectoris, identified pyrazolopyrimidine inhibitors of phosphodiesterase-5 (PDE5); the compound UK-92,480 (sildenafil) was synthesised in 1989.
- **Early 1990s.** In early clinical studies for angina, the cardiovascular effect was disappointing, but volunteers reported erections — an unexpected effect explained by the same mechanism (Section 3.3.3) acting in penile smooth muscle.
- **27 March 1998.** FDA approval as **Viagra** for erectile dysfunction.
- **2005.** After evidence that PDE5 inhibition relaxes pulmonary arteries (and the SUPER-1 trial), approval by the FDA and EMA as **Revatio** for pulmonary arterial hypertension (Ghofrani, Osterloh & Grimminger 2006).
- **Lessons.** Serendipity from a side effect; same target, different tissues; different dose and regimen for each indication. In our Fdataset, sildenafil is linked to "Pulmonary hypertension, primary, 1".

#### 3.11.2 Thalidomide: tragedy → leprosy → multiple myeloma

- **1957.** Chemie Grünenthal markets thalidomide (Contergan) in West Germany as a sedative, later widely used against morning sickness in pregnancy.
- **1961.** Widukind Lenz in Germany and William McBride in Australia independently link it to severe limb malformations (phocomelia) and other birth defects; Lenz reports to the company on 15 November 1961, McBride's letter appears in *The Lancet* on 16 December 1961, and the drug is withdrawn in November–December 1961. More than **10,000 children** were affected worldwide.
- **United States.** FDA reviewer Frances Oldham Kelsey refused to approve the application for lack of safety evidence, so the US was largely spared (though more than 2.5 million tablets had been distributed to US doctors in "investigational" use). The **Kefauver–Harris Amendments (10 October 1962)** followed.
- **1964–1965.** Israeli dermatologist Jacob Sheskin, looking for a sedative for a patient with a painful inflammatory complication of leprosy, *erythema nodosum leprosum* (ENL), gave thalidomide and saw the lesions resolve; he published in 1965.
- **16 July 1998.** FDA approval (Thalomid) for ENL — under a strict risk-management programme to prevent foetal exposure.
- **2006.** FDA approval, with dexamethasone, for **multiple myeloma**, after evidence of anti-myeloma activity in the late 1990s.
- **2010 onward.** Mechanism: cereblon (Ito et al. 2010); degradation of IKZF1/3 by lenalidomide (2014); SALL4 degradation linked to teratogenicity (Donovan et al. 2018).
- **Lessons.** A drug's biology does not disappear when it is withdrawn; a clinical observation made under very different circumstances led to new uses; **safety can be managed** for the right indication. Our benchmark links thalidomide to "Leprosy, susceptibility to, 3" and "Multiple myeloma".

#### 3.11.3 Minoxidil: blood pressure → hair loss

- Developed by Upjohn as an oral antihypertensive; approved by the FDA in **1979** (Loniten) for severe hypertension. Patients developed **hypertrichosis** (excess hair growth).
- **17 August 1988.** FDA approves 2 % topical minoxidil (**Rogaine**) for male-pattern baldness — the first FDA-approved drug for hair loss.
- **Lessons.** Side effect → new indication; **new route** (topical) to get local benefit with less systemic effect; prodrug activation (minoxidil sulfate) in the target tissue.

#### 3.11.4 Aspirin: pain and fever → heart attack prevention → (contested) cancer prevention

- **1897–1899.** Felix Hoffmann at Bayer synthesises acetylsalicylic acid; Bayer registers the trade name Aspirin on 6 March 1899. For decades it is an analgesic, antipyretic and anti-inflammatory.
- **1971.** John Vane shows that aspirin-like drugs inhibit prostaglandin synthesis (Nobel Prize 1982, shared with Bergström and Samuelsson). In platelets, which cannot make new proteins, aspirin's irreversible acetylation of cyclo-oxygenase-1 blocks thromboxane A<sub>2</sub> for the platelet's lifetime → reduced clotting. Low-dose aspirin becomes a cornerstone of secondary prevention of heart attack and stroke.
- **Cancer.** Rothwell et al. (2010), following 14,033 participants of five randomised cardiovascular trials for 20 years, found reduced colon-cancer incidence (HR 0.76) and mortality (HR 0.65), with a lag of 7–8 years. The US Preventive Services Task Force recommended low-dose aspirin for CVD *and colorectal cancer* prevention in some adults aged 50–59 in 2016 — and **withdrew** that recommendation in 2022, concluding the evidence that aspirin reduces colorectal cancer incidence or mortality was inadequate, given newer trials.
- **Lessons.** Mechanistic understanding (decades after first use) opened a new indication; **evidence can be revised** — a repositioning claim is never "finished".

#### 3.11.5 Metformin in oncology: a strong hypothesis that failed its definitive trial

- Metformin was introduced in France (1957–59) and the UK (1958), and approved by the FDA at the end of 1994 (marketed 1995). It is first-line therapy for type 2 diabetes.
- **2005.** Evans et al. (*BMJ*), using linked records from Tayside, Scotland, found that diabetics taking metformin had a lower risk of cancer, with a dose–response trend. Laboratory work suggested mechanisms (lower insulin, AMPK/mTOR effects). Hundreds of trials followed.
- **2022.** The MA.32 phase III trial (Goodwin et al., *JAMA* 2022) randomised **3,649** patients with high-risk early breast cancer *without diabetes* to metformin or placebo for 5 years, on top of standard therapy. Result: **no improvement** in invasive disease-free survival (hazard ratio 1.01).
- **Lessons.** Observational associations are hypotheses: people prescribed metformin differ from those who are not (confounding, "healthy user" effects, time-related biases). Only randomised trials can confirm efficacy. This is *exactly* the epistemic status of a model's prediction.

#### 3.11.6 Baricitinib: rheumatoid arthritis → COVID-19, found by a knowledge graph

- Baricitinib is an oral inhibitor of Janus kinases JAK1/JAK2, approved for rheumatoid arthritis (FDA, 2018).
- **February 2020.** Researchers at BenevolentAI, using a biomedical **knowledge graph** plus expert review, proposed baricitinib for COVID-19 (Richardson et al., *The Lancet*, 2020): it was predicted to inhibit AAK1 (a regulator of the endocytosis that viruses use to enter cells) at clinically achievable plasma concentrations, *and* its JAK inhibition could dampen the inflammatory response.
- **Trials.** ACTT-2 (Kalil et al., *NEJM* 2021; 1,033 patients): baricitinib plus remdesivir shortened median recovery from 8 to 7 days versus remdesivir alone. COV-BARRIER (Marconi et al., *Lancet Respir Med* 2021): 28-day mortality 8 % vs 13 % with placebo (a 38 % relative reduction).
- **Regulation.** FDA Emergency Use Authorization on **19 November 2020** (with remdesivir); full approval on **10 May 2022** for hospitalised adults needing oxygen or ventilation — the first immunomodulatory COVID-19 treatment to receive full FDA approval.
- **Lessons.** The best-known *computational* repositioning success — but note the path: computation → expert scrutiny of PK (plasma levels) → large randomised trials → regulatory approval. The algorithm produced the hypothesis; trials made it a treatment.

#### 3.11.7 Shorter cases worth knowing

| Drug | Original use | New use | How found | Key dates |
|---|---|---|---|---|
| Zidovudine (AZT) | Failed anticancer compound (synthesised 1964) | HIV/AIDS | Systematic screening of existing compounds | FDA approval 20 March 1987, the first antiretroviral |
| Amantadine | Influenza A antiviral (approved 1966) | Parkinson's disease | Clinical observation in a patient (Schwab et al., *JAMA* 1969) | Became an established Parkinson's treatment in the 1970s |
| Propranolol | Beta-blocker for heart disease and hypertension | Infantile haemangioma | Serendipity in infants treated for heart problems (Léauté-Labrèze et al., *NEJM* 2008) | Paediatric solution (Hemangeol) FDA-approved 14 March 2014 |
| Raloxifene | Osteoporosis (FDA 1997) | Reduction of invasive breast cancer risk | Secondary outcome in osteoporosis trial (MORE), then STAR trial | FDA approval 13 September 2007 |
| Imatinib | Chronic myeloid leukaemia (BCR-ABL; May 2001) | Gastrointestinal stromal tumour (KIT) | Rational: shared target (polypharmacology) | GIST approval February 2002 |
| Dexamethasone | Anti-inflammatory corticosteroid | Severe COVID-19 | Large pragmatic RCT (RECOVERY) | Announced 16 June 2020; final report (*NEJM* 2021): 28-day mortality 29.3 % vs 41.4 % in ventilated patients |
| Hydroxychloroquine | Malaria, lupus, rheumatoid arthritis | COVID-19 (**failed**) | In vitro antiviral activity, small uncontrolled studies | FDA EUA March 2020, **revoked 15 June 2020** after a large RCT showed no benefit and cardiac risks |

The last row is the cautionary tale: a plausible repurposing hypothesis, amplified before rigorous testing, was widely prescribed, and randomised trials showed no benefit. (Hydroxychloroquine is also a reminder that a drug can have *both* a failed and a valid repurposing hypothesis: Cheng et al. 2018 found, via network proximity plus patient-record analysis, an association with lower coronary artery disease risk — still a hypothesis.)

#### 3.11.8 Patterns across the histories

```
 HOW THE NEW USE WAS FOUND            EXAMPLES
 -----------------------------------  -----------------------------------------------
 side effect noticed in trials/use    sildenafil, minoxidil, amantadine, propranolol
 clinical intuition in a new setting  thalidomide (ENL)
 mechanism understood later           aspirin (antiplatelet)
 shared target / polypharmacology     imatinib (KIT), sildenafil (PDE5 in lung)
 epidemiology (observational)         metformin-cancer (failed), aspirin-cancer (contested)
 systematic screening                 zidovudine
 computation (knowledge graph)        baricitinib
 large pragmatic RCT of candidates    dexamethasone (RECOVERY)
```

Most historical successes were **serendipitous**. Computational repositioning aims to make serendipity systematic: to scan *all* drug–disease pairs for the patterns that, in hindsight, would have pointed to these discoveries.

### 3.12 Computational repositioning approaches

All approaches turn data about drugs, diseases and biology into a score for each (drug, disease) pair. They differ in *which* data and *what assumption* links data to therapy.

#### 3.12.1 Signature-based (transcriptomic) methods

**Idea.** Measure how a disease changes gene expression (genes up and down in diseased vs healthy tissue: the *disease signature*) and how each drug changes gene expression in cells (the *drug signature*). If a drug's signature is the **reverse** of the disease's, the drug might push the cells back towards health.

- **Connectivity Map (CMap)** (Lamb et al., *Science* 2006): a reference collection of expression profiles from human cell lines treated with small molecules, plus pattern-matching to query them; later scaled up as LINCS L1000.
- **Example.** Sirota et al. (2011) compared 100 disease signatures with 164 drug signatures and predicted that the anti-ulcer drug **cimetidine** could treat lung adenocarcinoma, confirmed in mouse xenografts. An independent replication (Kandela et al., *eLife* 2017) found the same direction of effect (tumours grew 2.37-fold vs 3.29-fold), but the difference was not statistically significant after correction for multiple comparisons. Even published preclinical validation is uncertain.
- **Assumption.** Expression reversal implies therapeutic effect; cell lines represent patient tissue. (The project lists LINCS L1000 as future work.)

#### 3.12.2 Target- and structure-based methods

**Idea.** Predict new *targets* for known drugs, then link targets to diseases.

- **Ligand-based**: compare a drug with the known ligands of each target. The Similarity Ensemble Approach (Keiser et al., *Nature* 2009) compared 3,665 drugs with ligand sets for 246 targets and experimentally confirmed 23 new drug–target associations.
- **Structure-based**: docking drugs into 3-D protein structures.
- **Assumption.** Binding implies a useful effect, and the target matters in the disease.

#### 3.12.3 Network-based methods

**Idea.** Place drug targets and disease genes on the **human protein–protein interaction network (interactome)**. A drug is a candidate for a disease if its targets lie *close* to the disease's genes (its *disease module*).

- **Network proximity** (Guney et al., *Nat Commun* 2016) measures the average shortest distance between drug targets and disease proteins, compared with random expectations.
- Cheng et al. (*Nat Commun* 2018) applied it to >900 drugs and then **tested predictions in patient records** with propensity-score matching (e.g. hydroxychloroquine and lower coronary artery disease risk), plus in-vitro experiments.
- **Assumption.** Network neighbourhood implies functional relevance; the interactome is complete enough (it is not).

#### 3.12.4 Machine learning on similarity matrices (this project's family)

**Idea.** Treat repositioning as **link prediction** or **recommendation** on a drug–disease matrix, using drug–drug and disease–disease similarity matrices as side information.

- **PREDICT** (Gottlieb et al., *Mol Syst Biol* 2011) built features from several drug similarities (chemical, side effects, targets) and disease similarities (phenotype, genetics), and trained a classifier on known indications. **Fdataset comes from this work.**
- **MBiRW** (Luo et al., *Bioinformatics* 2016) — bi-random walks on drug and disease similarity networks. **Cdataset comes from this work.**
- **Matrix factorisation / completion**: SCMFDD, DRRS (Unit B4).
- **Graph neural networks**: NIMCGCN, LAGCN, and our **MV-HGAT** (Units C3–C6).
- **Assumption.** Guilt by association (Section 3.13): similar drugs treat similar diseases.

#### 3.12.5 Knowledge graphs

**Idea.** Integrate many entity types (drugs, diseases, genes, pathways, side effects, anatomy…) and relation types into one heterogeneous graph; learn to predict "treats" edges.

- **Hetionet** (Himmelstein et al., *eLife* 2017): 47,031 nodes of 11 types and 2,250,197 relationships of 24 types, used to prioritise drug repurposing with meta-path features.
- **BenevolentAI's** knowledge graph and the baricitinib prediction (Section 3.11.6).
- **TxGNN** (Huang et al., *Nature Medicine* 2024): a graph foundation model trained on a medical knowledge graph to rank drugs for 17,080 diseases, including diseases with no existing treatments ("zero-shot"), with explanations based on multi-hop paths.
- Our project's heterogeneous graph (drug, disease and gene information with several relation types) is a small, carefully controlled knowledge graph.

#### 3.12.6 Real-world data and literature mining

- **Electronic health records and claims data**: look for drugs associated with better outcomes in a disease (Evans 2005 for metformin; Cheng 2018 for hydroxychloroquine). Powerful for *validating* hypotheses, but confounded.
- **Literature mining and language models**: extract drug–disease relations from papers. Risk: they rediscover what is already published and inherit publication bias.

#### 3.12.7 Comparison

| Approach | Main data | Core assumption | Handles new diseases? | Typical weakness |
|---|---|---|---|---|
| Signature | expression profiles | reversal ⇒ therapy | yes, if a signature exists | cell lines ≠ patients; noisy |
| Target/structure | ligands, protein structures | binding ⇒ effect | if disease targets known | binding ≠ efficacy |
| Network | interactome, disease genes | proximity ⇒ relevance | if disease genes known | incomplete interactome |
| Similarity ML (ours) | similarity matrices, known links | guilt by association | only via disease similarity (cold start) | rediscovers "more of the same" |
| Knowledge graph | many integrated sources | multi-hop paths ⇒ relevance | partly | data leakage, popularity bias |
| Real-world data | EHR, claims | association ⇒ effect | no | confounding |

### 3.13 Guilt by association

#### 3.13.1 The principle

**Guilt by association (GBA)**: entities that are similar tend to share properties. For repositioning:

- **Drug side:** if drug $i$ is similar to drug $i'$, and $i'$ treats disease $j$, then $i$ may treat $j$.
- **Disease side:** if disease $j$ is similar to disease $j'$, and drug $i$ treats $j'$, then $i$ may treat $j$.

Written as a score with a row-normalised similarity matrix $W$ (Section 4.1 computes it by hand):

$$\hat S^{\text{drug}}_{ij} = \sum_{i'} W_{ii'}\,A_{i'j}, \qquad \hat S^{\text{disease}}_{ij} = \sum_{j'} A_{ij'}\,W'_{jj'}.$$

In chemistry this is the **similar property principle** (structurally similar molecules tend to have similar properties). MV-HGAT's *multi-view propagation head* is exactly these formulas, one per similarity view, with learned weights (`HOW_IT_WORKS.md`, Section 4.3).

#### 3.13.2 Why it works (the biology)

- **Similar structure → similar targets.** Molecules that share substructures often bind the same protein pockets (e.g. the PDE5 inhibitors sildenafil and vardenafil share a scaffold; ECFP4 Tanimoto 0.59).
- **Similar targets → similar effects.** Drugs hitting the same protein or pathway produce related physiological effects.
- **Similar diseases → shared mechanisms.** Diseases with similar symptoms, genes or ontology position often share pathways (several types of pulmonary hypertension respond to the same vasodilators).
- **Therapeutic classes.** Medicine organises drugs into classes (statins, beta-blockers, anthracyclines), and new class members are usually tried for the class's indications.

**Evidence from our own data (Section 4.3).** For all 166,753 Fdataset drug pairs with structures, the probability that two drugs share at least one indication rises steeply with ECFP4 similarity:

| Tanimoto | pairs | share ≥ 1 indication | mean indication Jaccard |
|---|---|---|---|
| [0.0, 0.2) | 161,663 | 6.4 % | 0.019 |
| [0.2, 0.4) | 4,574 | 18.3 % | 0.078 |
| [0.4, 0.6) | 373 | 56.8 % | 0.300 |
| [0.6, 0.8) | 114 | 64.9 % | 0.295 |
| [0.8, 1.0] | 29 | 82.8 % | 0.399 |

GBA is real, and strong — *on average*.

#### 3.13.3 When it fails

Each failure mode below is a reason why a high-scoring prediction can be wrong, and several are visible in our own data.

1. **Activity cliffs: small structural change, large activity change.** Martin, Kofron & Traphagen (2002) found that a compound with Tanimoto ≥ 0.85 to an active compound had only about a **30 %** chance of being active itself in follow-up assays. Similarity is a probabilistic hint, not a guarantee.
2. **Stereoisomers look identical to 2-D fingerprints.** Quinine (antimalarial) and quinidine (antiarrhythmic) are stereoisomers with the same atoms and bonds; ECFP4 Tanimoto = **1.00** in our data, yet in Fdataset they share **no** indication. Likewise the R- and S-enantiomers of thalidomide have Tanimoto 1.000 without stereochemistry (0.714 when chirality is encoded) — and they interconvert in the body anyway. Retinoids (alitretinoin, tretinoin, isotretinoin; Tanimoto 1.00) have different indications in the benchmark too.
3. **Same target, opposite clinical effect, because of dosing.** Leuprolide and goserelin are synthetic analogues of gonadotropin-releasing hormone (GnRH; ECFP4 0.88 and 0.80 to natural "LHRH"). Natural GnRH given in *pulses* stimulates the reproductive axis (it treats hypogonadotropic hypogonadism); the analogues given *continuously* desensitise the pituitary and **suppress** sex hormones (they treat prostate and breast cancer, endometriosis). Same receptor, opposite outcome — a pharmacokinetic/pharmacodynamic fact that no similarity matrix contains.
4. **Different chemistry, same target.** Sildenafil and tadalafil are both PDE5 inhibitors used for erectile dysfunction and pulmonary hypertension, but their ECFP4 Tanimoto is only **0.115**. A chemistry-only GBA misses the relationship; in Fdataset, sildenafil's three nearest ECFP neighbours are the antipsychotics thiothixene and clozapine and the opioid meperidine (Exercise 10). Target-based views catch what chemistry misses — hence the value of multiple views.
5. **Route, formulation and tissue.** Oral and topical minoxidil, systemic and local drugs, drugs that do or do not cross the blood–brain barrier: GBA ignores PK.
6. **Disease heterogeneity.** "Breast cancer" is many diseases (hormone-receptor-positive, HER2-positive, triple-negative). A drug that works in one subtype can fail in another; disease similarity at the level of a single OMIM entry cannot see this.
7. **Label artefacts and biases.** The benchmark's indications were mapped to OMIM entries, which are designed for *genetic* diseases. Common conditions are mapped to the nearest OMIM entry, sometimes oddly: in Fdataset, minoxidil is linked to "Alopecia-epilepsy-pyorrhea-intellectual disability syndrome" (presumably via the word *alopecia*), and erectile dysfunction has no entry at all. GBA faithfully propagates such artefacts.
8. **Popularity and literature bias.** Well-studied drugs have more known indications and more genes in databases like CTD; GBA scores favour them ("the rich get richer"). Predictions for well-studied cancers (prostate cancer 10/10 "supported") may reflect how much has been *tried*, not how much *works*.
9. **GBA is weaker than it looks in large biological networks.** Gillis & Pavlidis (2012) showed that, in gene networks, the predictive signal is concentrated in a few highly informative connections and in "multifunctional" hub genes; most of the network contributes little. Average GBA performance can hide that predictions for a specific item rest on very little evidence.
10. **Circularity and leakage.** If a "similarity" was computed from the indications themselves (e.g. two diseases are similar because they share drugs), GBA re-discovers the labels. The project avoids similarities derived from the association matrix and excludes CTD's inferred and therapeutic gene–disease links for exactly this reason (`HOW_IT_WORKS.md`, Section 6; Unit E1).
11. **"More of the same".** GBA can only propose drugs resembling existing treatments. Truly novel mechanisms (the first drug for a disease) are, by construction, invisible to it — a fundamental limit in cold start.

### 3.14 From prediction to approved use: the validation ladder

Turning a top-10 prediction into a real repurposed drug requires climbing a ladder; each rung removes candidates.

```
 RUNG  ACTIVITY                                             QUESTION ANSWERED
 ----  ---------------------------------------------------  -----------------------------------------
  0    model ranks drug i for disease j                     is there a statistical pattern?
  1    retrospective validation (CV, LODO, cross-dataset)   does the method recover known links?
  2    evidence triage: literature, CTD, trial registries   has anyone seen this before? (audit it!)
  3    mechanistic plausibility                              target expressed in disease tissue? pathway?
  4    pharmacological feasibility                           dose, PK, tissue/BBB exposure, safety in
                                                              this population, contraindications
  5    in vitro experiments                                  effect in relevant cells?
  6    in vivo animal models                                 effect in a living organism? dose-response?
  7    real-world data (EHR/claims, propensity matching)     association in patients? (confounded)
  8    Phase II proof-of-concept RCT                         signal of efficacy in patients?
  9    Phase III confirmatory RCT(s)                         definitive efficacy and safety?
 10    regulatory submission (sNDA / 505(b)(2) / EMA)        label change: an approved indication
 11    Phase IV, guidelines, reimbursement, uptake           does it help patients in practice?
```

Alongside the science: **intellectual property and funding** (who pays for Phase II/III of an off-patent drug?), **ethics approval** for human studies, and **regulatory strategy** (orphan designation for rare diseases can make a programme viable). Many promising candidates stop at rung 2–4 for non-scientific reasons, and most of those that reach rung 8 fail there (Section 3.8).

### 3.15 Why a computational prediction is a hypothesis, not a recommendation

A score from MV-HGAT, MBiRW or any other model is **not** evidence that a drug is safe and effective for a disease. It is an **ordered list of hypotheses worth investigating**. The reasons, collected from this chapter and Unit B2:

1. **The labels are incomplete and noisy.** "Unknown = negative" (B2 Section 3.13), OMIM mapping artefacts, historical indications (old hormonal therapies), off-label uses of varying quality.
2. **Association is not causation.** GBA, network proximity and observational data detect patterns, not effects. Metformin's observational anticancer signal failed a 3,649-patient RCT.
3. **The model knows nothing about dose, PK, tissue exposure, contraindications, drug interactions or patient population.** Testosterone scores highly for prostate cancer in our case study, yet testosterone therapy is contraindicated in men with prostate cancer (Section 5.5).
4. **Scores are not calibrated probabilities** (B2 Section 3.4.3): a score of 0.99 does not mean 99 % likely.
5. **Retrospective metrics measure re-discovery, not discovery.** An AUC of 0.94 says the model recovers *hidden known* links; it does not measure how often a *new* prediction will work in patients.
6. **External "support" can be spurious.** Our own audit found that "49 trials" for amantadine in Alzheimer's were memantine trials.
7. **Most candidates fail in the clinic even after good preclinical evidence** (≈ 86 % failure from Phase I; 52 % of late failures for lack of efficacy).
8. **Real harm is possible.** Hydroxychloroquine for COVID-19 showed how premature belief in a repurposing hypothesis leads to widespread unproven use and adverse events.

**How to write about predictions.** Prefer "the model prioritises X as a candidate for Y; this hypothesis is supported by [specific, audited evidence] and requires experimental and clinical validation" over "X can treat Y". Never present predictions to patients or clinicians as treatment advice.

---

## 4. Code: pharmacology ideas you can compute

All blocks were run with the project's environment (`.venv\Scripts\python.exe`, CPU only); outputs are real. Blocks 4.3–4.5 read project files read-only; block 4.5 queries the public ClinicalTrials.gov API, so its numbers can change as the registry is updated (these were obtained on 1 October 2026).

### 4.1 Guilt by association on a toy matrix

Five drugs, four diseases. Each drug's score for a disease is the similarity-weighted vote of the *other* drugs that treat it.

```python
import numpy as np
import pandas as pd

drugs = ["drugA", "drugB", "drugC", "drugD", "drugE"]
diseases = ["angina", "hypertension", "pulm_hypertension", "migraine"]
# Known indications (1 = approved use, 0 = unknown)
A = np.array([[1, 1, 0, 0],    # drugA
              [1, 0, 1, 0],    # drugB
              [0, 1, 0, 0],    # drugC
              [0, 0, 0, 1],    # drugD
              [0, 0, 0, 0]])   # drugE: a drug with no known indication in this table
# Drug-drug similarity (symmetric, e.g. Tanimoto of fingerprints); diagonal ignored
S = np.array([[1.0, 0.8, 0.6, 0.1, 0.2],
              [0.8, 1.0, 0.5, 0.1, 0.7],
              [0.6, 0.5, 1.0, 0.2, 0.1],
              [0.1, 0.1, 0.2, 1.0, 0.3],
              [0.2, 0.7, 0.1, 0.3, 1.0]])
W = S - np.eye(len(S))                         # drop self-similarity
W = W / W.sum(axis=1, keepdims=True)           # each row sums to 1
score = W @ A                                  # weighted vote of the neighbours
df = pd.DataFrame(score, index=drugs, columns=diseases).round(3)
print(df)
# Rank only the UNKNOWN pairs: these are the repositioning hypotheses
cand = [(drugs[i], diseases[j], score[i, j]) for i in range(5) for j in range(4) if A[i, j] == 0]
cand.sort(key=lambda t: -t[2])
print("\nTop-3 hypotheses:")
for d, dz, sc in cand[:3]:
    print(f"  {d} -> {dz}: {sc:.3f}")
```

Output:

```text
       angina  hypertension  pulm_hypertension  migraine
drugA   0.471         0.353              0.471     0.059
drugB   0.381         0.619              0.000     0.048
drugC   0.786         0.429              0.357     0.143
drugD   0.286         0.429              0.143     0.000
drugE   0.692         0.231              0.538     0.231

Top-3 hypotheses:
  drugC -> angina: 0.786
  drugE -> angina: 0.692
  drugB -> hypertension: 0.619
```

Hand check for drugC → angina: drugC's similarities to the others are 0.6, 0.5, 0.2, 0.1 (sum 1.4); among them, drugA (0.6) and drugB (0.5) treat angina, so the score is $(0.6 + 0.5)/1.4 = 0.786$. drugE has no known indications at all, yet gets hypotheses through its similarity to drugB (0.7) — this is how GBA handles a "cold" drug that has side information. Notice also that known indications get high scores too (drugA → angina 0.471): when ranking candidates we exclude pairs already known.

### 4.2 When chemistry agrees and disagrees with pharmacology (RDKit)

ECFP4-like Morgan fingerprints (radius 2, 2,048 bits, as in the project's `chem_ecfp` view) and Tanimoto similarity for drugs discussed in this chapter. The SMILES are PubChem's.

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
smi = {
 "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
 "vardenafil": "CCCC1=NC(=C2N1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)CC)OCC)C",
 "tadalafil":  "CN1CC(=O)N2[C@@H](C1=O)CC3=C([C@H]2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36",
 "amantadine": "C1C2CC3CC1CC(C2)(C3)N",
 "memantine":  "CC12CC3CC(C1)(CC(C3)(C2)N)C",
 "R-thalidomide": "C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O",
 "S-thalidomide": "C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O",
}
mols = {k: Chem.MolFromSmiles(v) for k, v in smi.items()}
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
genc = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
fp = {k: gen.GetFingerprint(m) for k, m in mols.items()}
fpc = {k: genc.GetFingerprint(m) for k, m in mols.items()}
def t(a, b, f=fp): return DataStructs.TanimotoSimilarity(f[a], f[b])
for a, b in [("sildenafil","vardenafil"),("sildenafil","tadalafil"),("vardenafil","tadalafil"),("amantadine","memantine")]:
    print(f"{a:>13} vs {b:<13} Tanimoto = {t(a,b):.3f}")
print(f"R- vs S-thalidomide, no chirality:   {t('R-thalidomide','S-thalidomide'):.3f}")
print(f"R- vs S-thalidomide, with chirality: {t('R-thalidomide','S-thalidomide', fpc):.3f}")
```

Output:

```text
   sildenafil vs vardenafil    Tanimoto = 0.590
   sildenafil vs tadalafil     Tanimoto = 0.115
   vardenafil vs tadalafil     Tanimoto = 0.103
   amantadine vs memantine     Tanimoto = 0.455
R- vs S-thalidomide, no chirality:   1.000
R- vs S-thalidomide, with chirality: 0.714
```

- Sildenafil and vardenafil (same target, shared scaffold): 0.59 — GBA works.
- Sildenafil and tadalafil (same target, different scaffold): 0.115 — chemistry-only GBA misses a true pharmacological relationship.
- Amantadine and memantine: 0.455 in ECFP4 (and 1.000 in the benchmark's CDK fingerprint view) — the reason the model proposes amantadine for Alzheimer's disease.
- The two thalidomide enantiomers are indistinguishable without stereochemistry (1.000).

### 4.3 Is guilt by association true in our data?

For every pair of Fdataset drugs with structures, compare chemical similarity with shared indications.

```python
import numpy as np

z = np.load(r"C:/Users/Abhineet Anand/Desktop/DrugRepositioning/data/processed/Fdataset.npz")
A = z["A"].astype(bool)
T = z["drug_views"][list(z["drug_view_names"]).index("chem_ecfp")]   # ECFP4 Tanimoto, NaN = no structure
iu = np.triu_indices(len(A), k=1)                       # every unordered drug pair once
t = T[iu]
inter = (A[iu[0]] & A[iu[1]]).sum(1)                    # shared indications per pair
union = (A[iu[0]] | A[iu[1]]).sum(1)
jac = inter / np.maximum(union, 1)
ok = ~np.isnan(t)
print(f"{ok.sum()} drug pairs with structures for both drugs")
print(" Tanimoto bin   #pairs  P(share >=1 indication)  mean indication Jaccard")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    m = ok & (t >= lo) & (t < hi)
    print(f"  [{lo:.1f}, {min(hi, 1):.1f}{')' if hi < 1 else ']'}   {m.sum():7d}        {np.mean(inter[m] > 0):.3f}"
          f"                  {jac[m].mean():.3f}")
hi_sim_no_share = ok & (t >= .8) & (inter == 0)
print(f"pairs with Tanimoto >= 0.8 that share NO indication: {hi_sim_no_share.sum()} "
      f"of {(ok & (t >= .8)).sum()}")
```

Output:

```text
166753 drug pairs with structures for both drugs
 Tanimoto bin   #pairs  P(share >=1 indication)  mean indication Jaccard
  [0.0, 0.2)    161663        0.064                  0.019
  [0.2, 0.4)      4574        0.183                  0.078
  [0.4, 0.6)       373        0.568                  0.300
  [0.6, 0.8)       114        0.649                  0.295
  [0.8, 1.0]        29        0.828                  0.399
pairs with Tanimoto >= 0.8 that share NO indication: 5 of 29
```

Which highly similar pairs share *no* indication? Each is a lesson in pharmacology:

```python
import numpy as np

z = np.load(r"C:/Users/Abhineet Anand/Desktop/DrugRepositioning/data/processed/Fdataset.npz")
A = z["A"].astype(bool)
drugs, diseases = z["drug_names"], z["disease_names"]
T = z["drug_views"][list(z["drug_view_names"]).index("chem_ecfp")]
iu = np.triu_indices(len(A), k=1)
t = T[iu]
shared = (A[iu[0]] & A[iu[1]]).sum(1)
for k in np.flatnonzero(~np.isnan(t) & (t >= 0.8) & (shared == 0)):
    i, j = iu[0][k], iu[1][k]
    print(f"{drugs[i]} / {drugs[j]}   Tanimoto {t[k]:.2f}")
    print(f"   {drugs[i]}: {'; '.join(map(str, diseases[A[i]][:2]))}")
    print(f"   {drugs[j]}: {'; '.join(map(str, diseases[A[j]][:2]))}")
```

Output:

```text
Leuprolide / Lhrh   Tanimoto 0.88
   Leuprolide: Endometriosis, susceptibility to, 1; Uterine leiomyoma
   Lhrh: Hypogonadotropic hypogonadism 7 with or without anosmia
Goserelin / Lhrh   Tanimoto 0.80
   Goserelin: Malignant tumor of breast; Endometriosis, susceptibility to, 1
   Lhrh: Hypogonadotropic hypogonadism 7 with or without anosmia
Quinine / Quinidine   Tanimoto 1.00
   Quinine: Acanthosis nigricans-insulin resistance-muscle cramps-acral enlargement syndrome
   Quinidine: Ventricular arrhythmias due to cardiac ryanodine receptor calcium release deficiency syndrome; Atrial fibrillation, familial, 3
9-Cis-Retinoic Acid / Retinoic Acid   Tanimoto 1.00
   9-Cis-Retinoic Acid: Kaposi sarcoma, susceptibility to
   Retinoic Acid: Hyperpigmentation of eyelid; Hyperpigmentation of Fuldauer and Kuijpers
9-Cis-Retinoic Acid / Isotretinoin   Tanimoto 1.00
   9-Cis-Retinoic Acid: Kaposi sarcoma, susceptibility to
   Isotretinoin: Acute myeloid leukemia; Pyogenic arthritis-pyoderma gangrenosum-acne syndrome
```

- **Leuprolide, goserelin vs natural GnRH ("Lhrh")**: same receptor; continuous superagonist dosing suppresses the hormonal axis (cancers, endometriosis), pulsatile natural hormone stimulates it (hypogonadism).
- **Quinine vs quinidine**: stereoisomers; antimalarial vs antiarrhythmic. (Quinine is mapped to an OMIM syndrome whose name contains "muscle cramps" — presumably a trace of its historical use for leg cramps, and another label artefact.)
- **Alitretinoin (9-cis-retinoic acid) vs tretinoin (all-trans, listed as "Retinoic Acid") vs isotretinoin (13-cis)**: geometric isomers of the same molecule (identical 2-D fingerprints) with different receptor preferences, formulations and labelled uses — in the benchmark: Kaposi sarcoma skin lesions; topical treatment of skin hyperpigmentation; severe acne and a leukaemia entry.

### 4.4 Reading the case-study results

```python
import pandas as pd

R = "C:/Users/Abhineet Anand/Desktop/DrugRepositioning/results/case_studies/"
cases = {"104300": "Alzheimer disease type 1", "114480": "Malignant tumor of breast",
         "176807": "Prostate cancer"}
for omim, name in cases.items():
    df = pd.read_csv(R + f"Cdataset_OMIM{omim}.csv")
    ctd = df.CTD_curated.fillna("-") != "-"
    other = df.known_in_Fdataset == "yes"
    trials = df.clinical_trials.fillna(0) > 0
    support = ctd | other | trials
    print(f"{name} (OMIM {omim}): {support.sum()}/10 supported "
          f"[CTD {ctd.sum()}, other benchmark {other.sum()}, trials>0 {trials.sum()}]; "
          f"CTD 'therapeutic' {df.CTD_curated.fillna('').str.contains('therapeutic').sum()}")
    print("   unsupported:", ", ".join(df.drug[~support]))
```

Output:

```text
Alzheimer disease type 1 (OMIM 104300): 5/10 supported [CTD 1, other benchmark 0, trials>0 4]; CTD 'therapeutic' 1
   unsupported: Dantrolene, Ropinirole, L-Isoleucine, (+-)-Ethopropazine, Miglustat
Malignant tumor of breast (OMIM 114480): 6/10 supported [CTD 4, other benchmark 0, trials>0 5]; CTD 'therapeutic' 4
   unsupported: Estrone, (+)-Vincristine, Norethindrone, Ethinylestradiol
Prostate cancer (OMIM 176807): 10/10 supported [CTD 8, other benchmark 0, trials>0 8]; CTD 'therapeutic' 7
   unsupported:
```

This reproduces the "independent support" counts printed by `06_case_study.py`. The next block audits one of the evidence sources.

### 4.5 Auditing the evidence: what did ClinicalTrials.gov actually match?

```python
import requests

def trials(condition, drug):
    """Return all studies ClinicalTrials.gov finds for (condition, intervention)."""
    studies, token = [], None
    while True:
        params = {"query.cond": condition, "query.intr": drug, "pageSize": 200,
                  "fields": "NCTId,InterventionName,InterventionOtherName"}
        if token:
            params["pageToken"] = token
        page = requests.get("https://clinicaltrials.gov/api/v2/studies", params=params, timeout=60).json()
        studies += page.get("studies", [])
        token = page.get("nextPageToken")
        if not token:
            return studies

def names_in(study):
    ivs = study["protocolSection"].get("armsInterventionsModule", {}).get("interventions", [])
    return " ".join(i.get("name", "") + " " + " ".join(i.get("otherNames", [])) for i in ivs).lower()

for cond, drug, other in [("Alzheimer disease", "Amantadine", "memantine"),
                          ("Malignant tumor of breast", "Daunorubicin", "doxorubicin"),
                          ("Prostate cancer", "Paclitaxel", None)]:
    st = trials(cond, drug)
    lit = sum(drug.lower() in names_in(s) for s in st)
    msg = f"{drug:12s} x {cond:26s}: {len(st):3d} hits; drug literally named in {lit:3d}"
    if other:
        msg += f"; '{other}' named in {sum(other in names_in(s) for s in st)}"
    print(msg)
```

Output:

```text
Amantadine   x Alzheimer disease         :  49 hits; drug literally named in   0; 'memantine' named in 48
Daunorubicin x Malignant tumor of breast :  34 hits; drug literally named in   2; 'doxorubicin' named in 25
Paclitaxel   x Prostate cancer           :  55 hits; drug literally named in  45
```

ClinicalTrials.gov expands a drug name to related MeSH terms. "Amantadine" pulls in memantine trials (memantine is a dimethyl derivative of amantadine and sits under it in MeSH); "daunorubicin" pulls in doxorubicin. For paclitaxel the expansion is mostly harmless (brand names such as Abraxane and related taxanes). **Lesson:** counting registry hits is not evidence until you have looked at what was matched.

### 4.6 Pharmacokinetics: half-life, accumulation and steady state

```python
import numpy as np

# One-compartment model, intravenous doses (illustrative numbers, not a specific drug)
Vd, CL = 50.0, 5.0                 # volume of distribution (L), clearance (L/h)
k = CL / Vd                        # elimination rate constant (1/h)
t_half = np.log(2) / k
print(f"k = {k:.3f} /h, half-life = {t_half:.2f} h")

dose, tau = 100.0, 8.0             # 100 mg every 8 h
R = 1 / (1 - np.exp(-k * tau))     # accumulation factor
c_ss_avg = dose / (CL * tau)       # average steady-state concentration (F = 1)
c_ss_max = dose / Vd * R
c_ss_min = c_ss_max * np.exp(-k * tau)
print(f"accumulation factor {R:.3f}; steady state: avg {c_ss_avg:.2f}, peak {c_ss_max:.2f}, "
      f"trough {c_ss_min:.2f} mg/L")
for n in (1, 2, 3, 4, 5):
    print(f"after {n} half-lives: {1 - 0.5 ** n:.1%} of steady state")
# Simulate 10 doses to confirm the closed-form peak
t = np.arange(0, 10 * tau, 0.01)
c = sum(dose / Vd * np.exp(-k * (t - m * tau)) * (t >= m * tau) for m in range(10))
print(f"simulated peak just after the 10th dose: {c.max():.2f} mg/L")
```

Output:

```text
k = 0.100 /h, half-life = 6.93 h
accumulation factor 1.816; steady state: avg 2.50, peak 3.63, trough 1.63 mg/L
after 1 half-lives: 50.0% of steady state
after 2 half-lives: 75.0% of steady state
after 3 half-lives: 87.5% of steady state
after 4 half-lives: 93.8% of steady state
after 5 half-lives: 96.9% of steady state
simulated peak just after the 10th dose: 3.63 mg/L
```

These are Worked Example 1's numbers (Section 3.5). The simulation sums the decaying contribution of each past dose and confirms the closed-form peak.

### 4.7 Attrition arithmetic

```python
# Phase-transition probabilities quoted on FDA's "Step 3: Clinical Research" page
p1_to_2, p2_to_3 = 0.70, 0.33
for p3_to_filing in (0.25, 0.30):
    p_end_of_p3 = p1_to_2 * p2_to_3 * p3_to_filing
    print(f"P(enter Phase I -> pass Phase III) with Phase III rate {p3_to_filing:.2f}: {p_end_of_p3:.3f}"
          f"  (about 1 in {1 / p_end_of_p3:.0f})")
# Why each success is so expensive: the cost of failures is carried by the winners
cost = {"I": 25, "II": 60, "III": 250}          # illustrative out-of-pocket $M per programme (NOT data)
p_reach = {"I": 1.0, "II": p1_to_2, "III": p1_to_2 * p2_to_3}
spend_per_start = sum(cost[ph] * p_reach[ph] for ph in cost)
p_success = p1_to_2 * p2_to_3 * 0.275
print(f"expected clinical spend per programme started: ${spend_per_start:.0f}M; "
      f"per success: ${spend_per_start / p_success:,.0f}M")
```

Output:

```text
P(enter Phase I -> pass Phase III) with Phase III rate 0.25: 0.058  (about 1 in 17)
P(enter Phase I -> pass Phase III) with Phase III rate 0.30: 0.069  (about 1 in 14)
expected clinical spend per programme started: $125M; per success: $1,964M
```

The first two lines use the FDA's published phase-transition rates. The cost lines use *invented* round numbers purely to show the logic — every approved drug carries the cost of the programmes that failed — which is why published per-approval estimates (US$1–2.6 billion) are so much larger than the cost of any single trial.

---

## 5. In this project: pharmacology inside the code and the results

### 5.1 What the association matrix really contains

`A[i, j] = 1` means "drug $i$ (a DrugBank ID) has a recorded indication for disease $j$ (an OMIM ID)" in the benchmark. Fdataset was assembled by Gottlieb et al. (2011) for PREDICT, Cdataset by Luo et al. (2016) for MBiRW. Three pharmacological facts about these labels shape everything downstream:

1. **They are indications mapped to OMIM.** OMIM catalogues mainly Mendelian (genetic) diseases, so common conditions are represented by the nearest OMIM entry, sometimes a rare syndrome or a "susceptibility to" entry ("Leprosy, susceptibility to, 3" for thalidomide; "Alopecia-epilepsy-pyorrhea-intellectual disability syndrome" for minoxidil). Conditions without a good OMIM entry (erectile dysfunction) are simply absent.
2. **They are a historical snapshot.** Fdataset dates from 2011. Historical uses (high-dose estrogens in breast cancer) and uses that are not approved indications (valproic acid and vitamin E appear among the "known" Alzheimer's drugs) are present; later approvals are not. Every 0 is "unknown", not "ineffective" (B2, Section 3.13).
3. **They include repositioned uses.** Thalidomide (leprosy, myeloma), sildenafil (pulmonary hypertension), minoxidil (alopecia), metformin (polycystic ovaries — an off-label/secondary use). Retrospective evaluation therefore partly measures how well a method *re-discovers past repositioning*.

### 5.2 The six similarity views, read as pharmacology

| View | What it measures | Pharmacological rationale | Typical failure |
|---|---|---|---|
| `chem_cdk`, `chem_ecfp` (drug) | Tanimoto similarity of 2-D fingerprints | similar property principle: similar structure → similar targets → similar effects | activity cliffs, stereoisomers, different scaffolds on one target (Section 3.13.3) |
| `gene_r` (drug) | Jaccard overlap of CTD chemical–gene interaction sets | shared targets / pathways / transcriptional effects | CTD interactions are mostly expression changes, not binding; literature bias (well-studied drugs have many genes) |
| `pheno_mim` (disease) | MimMiner text similarity of OMIM descriptions | similar symptoms suggest shared mechanisms and shared treatments | similar symptoms from different causes |
| `sem_mondo` (disease) | Wang semantic similarity in the MONDO ontology | nearby diseases in a classification often share pathophysiology | classification by organ or symptom, not mechanism |
| `gene_d` (disease) | Jaccard overlap of CTD curated disease genes | shared molecular mechanism | half the diseases have no curated genes; genetic association ≠ druggable mechanism |

A *target* view (Jaccard of DrugBank targets) would capture the sildenafil–tadalafil relationship that chemistry misses; `HOW_IT_WORKS.md` Section 9 describes how to add it if DrugBank access becomes available.

### 5.3 Guilt by association, literally: the propagation head

In `src/drepo/methods.py`, `MVHGATMethod.fit_predict` defines:

```python
def propagation(Am):
    """One (drugs x diseases) score slice per view, from VISIBLE links only."""
    if not c.prop_head:
        return None
    Am = Am.float()
    return torch.stack([K @ Am for K in self._prop["drug"]] +
                       [(K @ Am.T).T for K in self._prop["disease"]])
```

with `self._prop["drug"]` built by `knn_kernel(data.drug_view(v), c.k)` — a similarity matrix sparsified to each drug's $k = 10$ nearest neighbours. So `K @ Am` is exactly $\hat S^{\text{drug}} = W A$ of Section 3.13.1 (one slice per drug view) and `(K @ Am.T).T` is $\hat S^{\text{disease}} = A W'^{\top}$ (one slice per disease view). The model learns a non-negative weight per slice and adds them to the GNN's logit. In other words, MV-HGAT contains six explicit guilt-by-association scorers and learns how much to trust each. The ablation "no propagation head" (`04_ablation.py`) measures how much of the performance comes from this classical pharmacological heuristic.

### 5.4 `scripts/06_case_study.py`: the evidence types

```python
"CTD_curated": ctd_evidence(db, omim),
f"known_in_{other.name}": "yes" if (db, omim) in other_links else "-",
"clinical_trials": trials(name, dname),
...
confirmed = (df.CTD_curated != "-") | (df[f"known_in_{other.name}"] == "yes") | \
            (df.clinical_trials.fillna(0).astype(float) > 0)
```

Read each evidence type pharmacologically:

- **CTD curated "therapeutic"**: curators found literature that the chemical has a known or potential therapeutic role in the disease. The strongest of the three, but "potential" includes preclinical studies.
- **CTD curated "marker/mechanism"**: the chemical *correlates with* the disease or *may play a role in its cause* — for example, a hormone that drives a cancer. This is evidence of *biological relevance*, not of benefit, and sometimes evidence of **harm**.
- **Known in the other benchmark**: an independent curation of indications — good evidence, rarely triggered.
- **ClinicalTrials.gov count > 0**: someone registered a study. A trial is a hypothesis being tested, not a result; it may have failed; the drug may be a background therapy or a comparator; and the registry's synonym expansion can match the wrong drug (Section 4.5).

### 5.5 A pharmacologist reads our three case studies

**Prostate cancer (OMIM 176807) — "10/10 supported".** Methotrexate, doxorubicin, fulvestrant, paclitaxel, estrone, 5-fluorouracil, gemcitabine, sorafenib, epirubicin, testosterone. Most are cytotoxic chemotherapies used broadly across cancers; that many have prostate-cancer trials says as much about how often chemotherapies are *tried* across tumours as about efficacy. The model's evidence came mainly from disease-side views (`pheno_mim`, `gene_d`) — "drugs used for cancers that resemble prostate cancer", textbook disease-side GBA. Two entries need care:

- **Testosterone** (rank 10; CTD "marker/mechanism|therapeutic"; 125 trial hits, of which only 38 name testosterone as an intervention). Androgens drive most prostate cancers; androgen-deprivation therapy is standard treatment and testosterone products are *contraindicated* in men with known or suspected prostate cancer. Much of the registry matching reflects trials that *suppress or measure* testosterone. There is a genuine but niche research line (high-dose "bipolar androgen therapy" in castration-resistant disease), so the hypothesis is not absurd — but "supported" would mislead a reader.
- **Estrone** (marker/mechanism): estrogens were historically used to suppress testosterone in prostate cancer (the known drugs include estradiol, conjugated estrogens and estramustine), so its high score is GBA from those labels; "marker/mechanism" is not therapeutic evidence.

**Breast cancer (OMIM 114480) — 6/10.** Estrone (rank 1), medroxyprogesterone, ethinylestradiol and norethindrone are sex hormones. Breast cancer's known drugs in Cdataset include estradiol, conjugated estrogens and testosterone — remnants of the era when high-dose hormonal therapy was used for advanced breast cancer — so chemical GBA pulls in more hormones. But estrogens also *stimulate* hormone-receptor-positive breast cancer. A model that cannot distinguish agonists from antagonists, or old from current practice, will happily suggest both. Zoledronic acid (bone metastases), etoposide, irinotecan, carboplatin, vincristine, daunorubicin are chemotherapies with varying evidence; daunorubicin's 34 trial hits are mostly doxorubicin trials (only 2 name daunorubicin).

**Alzheimer's disease (OMIM 104300) — 5/10, really 3–4/10.** Amantadine (chemical similarity to memantine; its 49 "trials" are memantine trials), then several **dopaminergic Parkinson's drugs** (ropinirole, pramipexole, rasagiline, levodopa, entacapone, ethopropazine) and dantrolene, isoleucine, miglustat. Why Parkinson's drugs? The occlusion column says their main evidence is `assoc`, the known-association graph. Among Alzheimer's *known* drugs in Cdataset is **selegiline**, an MAO-B inhibitor for Parkinson's disease that was tested in Alzheimer's disease in the 1990s (together with vitamin E — which is why alpha-tocopherol is also a "known" Alzheimer's drug). Selegiline is also linked to several Parkinson's entries, as are ropinirole, levodopa and rasagiline (another MAO-B inhibitor), so collaborative "drugs that share diseases with Alzheimer's drugs" signals pull the Parkinson's pharmacopoeia in. Miglustat's evidence is instead `pheno_mim`: Alzheimer's closest phenotypic neighbours in MimMiner include Huntington disease and Niemann–Pick disease type C1, and miglustat is used for Niemann–Pick type C. Some candidates have genuinely been tested in Alzheimer's: we verified two completed Phase II trials of rasagiline (NCT00104273, NCT02359552) and one of R(+)-pramipexole (NCT01388478). Read the last one carefully: the marketed Parkinson's drug is the *S*-enantiomer of pramipexole, while R(+)-pramipexole (dexpramipexole) has little dopamine-agonist activity — pharmacologically a different drug, matched by name. Removing amantadine gives 4/10; also discounting the R(+)-pramipexole trial gives 3/10. The remaining supported hypotheses have *already* been explored — whether they worked is a separate question that a trial count does not answer.

**Global view attention** (`results/case_studies/Cdataset_*_view_attention.csv`): disease nodes put about 45 % of their attention on known associations and roughly 17–20 % on each of `pheno_mim`, `sem_mondo` and `gene_d`; drug nodes about 77 % on associations and ~8 % on each similarity view. Attention is not proof of causal importance (Unit E2), which is why the case-study tables also report occlusion.

### 5.6 How to present these results honestly

- Report precision@10 against external evidence **with the evidence type** ("therapeutic" vs "marker/mechanism"), and audit trial matches by intervention name.
- Report a **background rate**: what fraction of *random* drugs (or of a baseline method's top 10) would have "support" for the same disease? For prostate cancer, where chemotherapies are tried widely, the background may be high (Exercise 8).
- Separate "re-discoveries of practice that predates the benchmark" from "genuinely novel hypotheses".
- Use the wording of Section 3.15: candidates for investigation, not treatments.

---

## 6. Common mistakes and misconceptions

1. **"High model score = the drug works."** It is a ranked hypothesis; efficacy needs RCTs (Section 3.15).
2. **"Repurposing skips clinical trials."** It usually skips or shortens preclinical work and Phase I; efficacy trials (Phase II/III) are still required, and most failures are for lack of efficacy.
3. **"Same target ⇒ same indication."** Dose, regimen and tissue matter: GnRH agonists suppress or stimulate depending on dosing; topical vs oral minoxidil.
4. **"Similar structure ⇒ similar activity, always."** Only probabilistically (≈ 30 % for Tanimoto ≥ 0.85 in Martin et al. 2002); stereoisomers can have different uses.
5. **"Dissimilar structure ⇒ unrelated."** Sildenafil and tadalafil (Tanimoto 0.115) share target and indications.
6. **Treating CTD "marker/mechanism" as therapeutic evidence.** A hormone that drives a cancer is a "marker/mechanism" chemical for that cancer.
7. **Counting trial-registry hits without reading them.** Synonym expansion, background drugs, comparators, failed trials.
8. **Confusing observational association with causation.** Metformin–cancer looked compelling in observational data and failed in a 3,649-patient RCT.
9. **Ignoring pharmacokinetics.** Brain diseases need brain exposure; new populations change clearance; a higher dose re-opens safety.
10. **Confusing indication, off-label use and label.** Off-label use is legal prescribing practice, not approval; promotion of off-label use is restricted.
11. **Thinking generics make repurposing commercially easy.** Off-patent status *removes* the incentive to pay for trials.
12. **Assuming the benchmark is ground truth.** OMIM mapping artefacts and historical labels are baked into $A$.
13. **Equating "biologically relevant" with "beneficial".** Many relevant chemicals are harmful in the disease (estrogens in ER-positive breast cancer).
14. **Over-claiming from AI success stories.** Baricitinib succeeded because computation was followed by expert PK review and large RCTs.

---

## 7. Exercises

★ conceptual · ★★ quantitative · ★★★ coding / design.

**Exercise 1 (★).** Name the concept (indication, off-label use, contraindication, polypharmacology, or a repositioning type) in each case:
(a) A doctor prescribes propranolol to calm performance anxiety.
(b) Imatinib inhibits both BCR-ABL and KIT.
(c) Topical minoxidil is approved for hair loss.
(d) Testosterone products must not be used in men with suspected prostate cancer.
(e) Zidovudine, abandoned as an anticancer agent, becomes an HIV drug.
(f) Sildenafil's US label for Revatio lists pulmonary arterial hypertension.

<details><summary>Solution</summary>

(a) Off-label use (performance anxiety is not a labelled indication of propranolol). (b) Polypharmacology. (c) Repositioning via a new formulation/route (and, once approved, an indication of that product). (d) Contraindication. (e) Repositioning of a shelved/failed compound ("drug rescue"). (f) Indication — and an example of a repositioned indication.
</details>

**Exercise 2 (★★).** A drug follows the Hill model with $E_{\max} = 100\%$ and EC<sub>50</sub> = 10 nM. What concentration gives 80 % of the maximal effect if $n = 1$? If $n = 2$? What does the comparison tell you about steep dose–response curves?

<details><summary>Solution</summary>

$0.8 = C^n/(10^n + C^n) \Rightarrow C^n = 4\cdot10^n \Rightarrow C = 10\cdot 4^{1/n}$ nM. For $n=1$: 40 nM. For $n=2$: 20 nM. With a steeper curve ($n=2$) the effect rises from small to large over a narrow concentration range: small PK changes (a different population, an interacting drug) can switch the effect on or off — important when a repositioned use needs a different dose.
</details>

**Exercise 3 (★★).** A drug has $t_{1/2} = 12$ h and $V_d = 100$ L. (a) Compute $CL$. (b) With oral bioavailability $F = 0.5$, what average steady-state concentration does 200 mg every 12 h give? (c) How long until 90 % of steady state? (d) What loading dose (oral) would reach 1.44 mg/L immediately?

<details><summary>Solution</summary>

(a) $CL = 0.693\,V_d/t_{1/2} = 0.693\times100/12 = 5.78$ L/h.
(b) $\bar C_{ss} = F\cdot\text{Dose}/(CL\cdot\tau) = 0.5\times200/(5.78\times12) = 100/69.3 = 1.44$ mg/L.
(c) $1 - 2^{-n} = 0.9 \Rightarrow n = \log_2 10 = 3.32$ half-lives $= 39.9$ h.
(d) Loading dose $= C\cdot V_d/F = 1.44\times100/0.5 = 288$ mg.
</details>

**Exercise 4 (★★).** Suppose phase-transition probabilities are 0.52 (I→II), 0.29 (II→III), 0.58 (III→submission) and 0.90 (submission→approval). (a) What is the probability that a drug entering Phase I is approved? (b) How many Phase I entrants are needed on average per approval? (c) How many independent Phase I programmes give at least a 90 % chance of one or more approvals?

<details><summary>Solution</summary>

(a) $p = 0.52\times0.29\times0.58\times0.90 = 0.0787$ (7.9 %).
(b) $1/p = 12.7$.
(c) Need $1 - (1-p)^n \ge 0.9 \Rightarrow n \ge \ln 0.1/\ln(1-0.0787) = 2.3026/0.0820 = 28.1$, so **29** programmes. Attrition is why portfolios, not single projects, are the unit of drug development.
</details>

**Exercise 5 (★).** Give three reasons repositioning is faster and cheaper than *de novo* development, and the main scientific risk it does *not* remove. Name one non-scientific obstacle.

<details><summary>Solution</summary>

Faster/cheaper: known human safety and tolerability (skip or shorten Phase I), known PK and formulation, existing manufacturing and supply, possible use of prior data in a 505(b)(2) or supplemental application. Risk not removed: **efficacy** in the new disease (the main cause of Phase II/III failure; 52 % in Harrison 2016) — plus any safety issue introduced by a different dose or population. Non-scientific obstacle: weak commercial incentive for off-patent generics (anyone can sell the drug; off-label prescribing of cheap generics), so funding for confirmatory trials is hard to find.
</details>

**Exercise 6 (★★).** Disease-side GBA by hand. Disease D is similar to diseases D1 (similarity 0.6), D2 (0.3) and D3 (0.1). Drug X treats D1 and D3; drug Y treats D2; drug Z treats D1 and D2. Using row-normalised similarities, score X, Y and Z for D. Which would you propose first, and what would make you doubt it?

<details><summary>Solution</summary>

Weights: 0.6, 0.3, 0.1 (they already sum to 1). X: 0.6 + 0.1 = 0.7. Y: 0.3. Z: 0.6 + 0.3 = 0.9. Rank: Z, X, Y. Doubts: is D similar to D1/D2 for a *mechanistic* reason or only in symptoms? Does Z's mechanism make sense in D? Is Z's use in D1/D2 current practice or historical? Can Z reach the affected tissue at a tolerable dose? Is D a heterogeneous disease where Z might help only a subtype?
</details>

**Exercise 7 (★).** For each case history in Section 3.11, say which computational approach (Section 3.12) would most plausibly have found it in advance, if any.

<details><summary>Solution</summary>

- Sildenafil → PAH: target/network-based (PDE5 expressed in pulmonary vasculature; pathway relevance). → ED: hard to predict; found via a side effect (side-effect similarity methods could help).
- Thalidomide → ENL/myeloma: very hard; the mechanism (cereblon) was unknown until 2010. Perhaps signature-based (anti-inflammatory/TNF effects).
- Minoxidil → alopecia: side-effect-based methods (hypertrichosis listed as a side effect).
- Aspirin → CVD prevention: mechanism-based (platelet COX-1 → thromboxane); → colorectal cancer: real-world/RCT secondary analyses.
- Metformin → cancer: real-world data (and the RCT shows the risk of trusting it).
- Baricitinib → COVID-19: knowledge graph (actually found this way).
- Imatinib → GIST: target-based (KIT).
- Amantadine → Parkinson's: side-effect/clinical observation; today, also chemical/target similarity to dopaminergic and NMDA drugs.
- Dexamethasone → COVID-19: mechanism-based (anti-inflammatory) plus a platform RCT.
- Similarity-based ML (our family) excels at "more of the same" (e.g. new members of a drug class for the class's indications), and would struggle with genuinely first-in-class discoveries like thalidomide's.
</details>

**Exercise 8 (★★).** Suppose that for prostate cancer 30 % of drugs drawn at random from the benchmark would show "independent support" (a CTD link or any trial). What is the probability that a random list of 10 drugs gets 8 or more supported? What does that imply for interpreting "10/10"? How would you estimate the 30 % in practice?

<details><summary>Solution</summary>

$X \sim \text{Binomial}(10, 0.3)$: $P(X\ge8) = \binom{10}{8}0.3^8 0.7^2 + \binom{10}{9}0.3^9 0.7 + 0.3^{10} = 0.001447 + 0.000138 + 0.000006 = 0.00159$. So 10/10 would be very unlikely by chance *if* the background were 30 %. But if the background among *anticancer* drugs is, say, 80 %, then $P(X = 10) = 0.8^{10} = 0.107$ — not impressive. The right comparison is with a background drawn from comparable drugs (e.g. top-10 lists of a simple baseline, or random drugs from the same therapeutic class). Estimate it empirically: run the same evidence pipeline on many random drugs or on baseline methods' top-10 lists, with the same intervention-name audit.
</details>

**Exercise 9 (★★).** The prostate-cancer list contains testosterone with "marker/mechanism|therapeutic" CTD evidence and 125 registry hits. Write a short (≤ 5 sentences) assessment as a pharmacologist would.

<details><summary>Solution</summary>

Androgens drive most prostate cancers, and lowering testosterone is standard therapy; testosterone products are contraindicated in men with known or suspected prostate cancer. The "marker/mechanism" evidence reflects this causal role, not benefit, and most registry hits are trials that suppress or measure testosterone (only 38 of 125 list it as an intervention). A niche hypothesis exists (high-dose bipolar androgen therapy in castration-resistant disease), so the prediction is not meaningless, but it should not be counted as support for therapeutic use. The model likely scored it high through disease-side GBA from hormone therapies used for related cancers and from estrogens' historical role. Verdict: biologically relevant, therapeutically contraindicated in general use; flag rather than count.
</details>

**Exercise 10 (★★★, coding).** Hide the known link sildenafil → "Pulmonary hypertension, primary, 1" in Fdataset and ask whether simple drug-side (ECFP) and disease-side (MimMiner) GBA would rediscover it. Explain the result pharmacologically.

<details><summary>Solution</summary>

```python
import numpy as np
z = np.load(r"C:/Users/Abhineet Anand/Desktop/DrugRepositioning/data/processed/Fdataset.npz")
A = z["A"].astype(float); drugs = list(z["drug_names"]); dis = list(z["disease_names"])

def knn_weights(S, k=10):
    S = np.nan_to_num(S.copy()); np.fill_diagonal(S, 0)
    W = np.zeros_like(S); nn = np.argsort(-S, axis=1)[:, :k]
    r = np.arange(len(S))[:, None]; W[r, nn] = S[r, nn]
    return W / (W.sum(1, keepdims=True) + 1e-12)

Wr = knn_weights(z["drug_views"][list(z["drug_view_names"]).index("chem_ecfp")])
Wd = knn_weights(z["disease_views"][list(z["disease_view_names"]).index("pheno_mim")])
i, j = drugs.index("Sildenafil"), dis.index("Pulmonary hypertension, primary, 1")
print("known PAH drugs:", ", ".join(drugs[k] for k in np.flatnonzero(A[:, j])))
At = A.copy(); At[i, j] = 0                     # pretend we do not know sildenafil treats PAH
for name, S in [("drug side (ECFP)", Wr @ At), ("disease side (MimMiner)", At @ Wd.T)]:
    col = S[:, j][At[:, j] == 0]                # candidate drugs = the unknown ones
    s = S[i, j]
    print(f"{name:24s} sildenafil score {s:.3f}; {int((col > s).sum())} candidates score higher, "
          f"{int((col == s).sum()) - 1} tie with it ({len(col)} candidates)")
print("sildenafil's 3 nearest ECFP neighbours:", ", ".join(drugs[k] for k in np.argsort(-Wr[i])[:3]))
T = z["drug_views"][list(z["drug_view_names"]).index("chem_ecfp")]
print(f"ECFP Tanimoto sildenafil-tadalafil = {T[i, drugs.index('Tadalafil')]:.3f}")
```
```text
known PAH drugs: Sildenafil, Treprostinil, Bosentan, Tadalafil, Isosorbide Dinitrate, Iloprost, Prostacyclin
drug side (ECFP)         sildenafil score 0.000; 32 candidates score higher, 554 tie with it (587 candidates)
disease side (MimMiner)  sildenafil score 0.000; 50 candidates score higher, 536 tie with it (587 candidates)
sildenafil's 3 nearest ECFP neighbours: Thiothixene, Clozapine, Meperidine
ECFP Tanimoto sildenafil-tadalafil = 0.115
```

Both scorers give sildenafil 0 — no signal. Drug side: the only other PDE5 inhibitor in the PAH list, tadalafil, is chemically dissimilar (0.115), and sildenafil's nearest chemical neighbours are psychiatric/opioid drugs with no PAH link. Disease side: none of PAH's phenotypic neighbours is treated by sildenafil in the benchmark (its only other link is an odd renal OMIM entry; erectile dysfunction has no OMIM entry). The historical discovery depended on *target* biology (PDE5 in pulmonary arteries), which neither view encodes. A target-based view, or a learned model combining views, is needed — and even then, sildenafil's PAH use is a case where pharmacological reasoning beat similarity.
</details>

**Exercise 11 (★★★, coding).** On Cdataset, compute pure drug-side GBA scores for "Alzheimer disease type 1" using `chem_cdk` and `chem_ecfp` (10 nearest neighbours), list the top-5 new candidates, and compare with MV-HGAT's case-study list.

<details><summary>Solution</summary>

```python
import numpy as np

z = np.load(r"C:/Users/Abhineet Anand/Desktop/DrugRepositioning/data/processed/Cdataset.npz")
A = z["A"].astype(float); drugs = list(z["drug_names"]); dis = list(z["disease_names"])

def knn_weights(S, k=10):
    S = np.nan_to_num(S.copy()); np.fill_diagonal(S, 0)
    W = np.zeros_like(S); nn = np.argsort(-S, axis=1)[:, :k]
    r = np.arange(len(S))[:, None]; W[r, nn] = S[r, nn]
    return W / (W.sum(1, keepdims=True) + 1e-12)

j = dis.index("Alzheimer disease type 1")
for view in ["chem_cdk", "chem_ecfp"]:
    W = knn_weights(z["drug_views"][list(z["drug_view_names"]).index(view)])
    s = (W @ A)[:, j]
    s[A[:, j] == 1] = -np.inf                      # rank only drugs not already known
    top = np.argsort(-s)[:5]
    print(f"{view:9s} top-5 new candidates: " + ", ".join(f"{drugs[i]} ({s[i]:.2f})" for i in top))
```
```text
chem_cdk  top-5 new candidates: Amantadine (0.29), Tetrahydrocannabinol (0.21), Rasagiline (0.19), Mecamylamine (0.16), Azelaic Acid (0.16)
chem_ecfp top-5 new candidates: Amantadine (0.31), Tetrabenazine (0.20), Methadone (0.19), Disulfiram (0.18), Benzphetamine (0.16)
```

Amantadine tops both lists: MV-HGAT's first suggestion is reproduced by a one-line GBA rule, through its similarity to memantine. The rest of the chemical lists are noisy (azelaic acid, a skin drug; benzphetamine, an appetite suppressant), while MV-HGAT's list below amantadine is dominated by dopaminergic drugs coming from *disease-side* evidence (Section 5.5). Combining views changes which hypotheses rise — and none of them is validated by the ranking itself.
</details>

**Exercise 12 (★★★, design).** Draft a validation plan for "amantadine for Alzheimer's disease", from the current prediction to a decision about a Phase II trial. Be specific about what each step would check.

<details><summary>Solution</summary>

1. **Audit the evidence**: search literature and registries for amantadine *by name* in Alzheimer's/dementia (excluding memantine matches); note any completed studies and their results.
2. **Mechanistic rationale**: amantadine's weak NMDA antagonism (memantine's mechanism), dopaminergic effects; are these relevant to Alzheimer's cognition or to specific symptoms (apathy, agitation)? Compare receptor affinities with memantine's at clinical concentrations.
3. **Pharmacology and safety**: CNS penetration (yes — it acts centrally in Parkinson's); renal clearance (doses must be reduced in kidney impairment, common in the elderly); known adverse effects in older adults (confusion, hallucinations) — a serious concern in dementia; drug interactions with anticholinergics.
4. **Preclinical**: effects in Alzheimer's animal models on cognition and pathology, if not already published; dose–response.
5. **Real-world data**: in health records, do patients with Parkinson's or influenza exposure to amantadine show different dementia incidence? Use active-comparator designs and propensity scores; interpret cautiously.
6. **Decision**: given that memantine already exists, what would amantadine add (cost, a different symptom target)? If the case is strong, design a Phase II RCT (randomised, placebo-controlled, defined cognitive and safety endpoints, renal-dose adjustment), seek ethics approval and non-commercial funding.
7. **Expected outcome realistically**: many such hypotheses stop at step 1–3; the elderly-safety profile may already make this one unattractive.
</details>

**Exercise 13 (★).** The project excludes CTD *therapeutic* chemical–disease and gene–disease links from the model's inputs but uses CTD curated chemical–disease links to *validate* case studies. Explain why both choices are correct.

<details><summary>Solution</summary>

Therapeutic links encode the answer — that a chemical treats a disease. Feeding them (or gene–disease links inferred through them) to the model would leak labels into the inputs: the model would "predict" what it was told, inflating metrics and making case studies circular. As a validation source they are legitimate precisely because the model never saw them: agreement is independent evidence (subject to the caveats of Section 5.4).
</details>

**Exercise 14 (★★).** Explain, using the hydroxychloroquine and metformin stories, why a correlation found in observational data (or a high model score) can fail to translate into a treatment effect. Name two specific biases.

<details><summary>Solution</summary>

Observational associations can arise without causation. **Confounding by indication / healthy-user effects**: patients prescribed metformin differ systematically (e.g. earlier-stage diabetes, better access to care) from those on other drugs; **immortal-time and time-window biases**: misclassifying follow-up time before treatment start can make a drug look protective. For hydroxychloroquine, early small, uncontrolled studies and in-vitro activity did not account for disease course, patient selection, or the concentrations achievable safely in humans; randomised trials showed no benefit and cardiac risk. A model score inherits similar problems: it reflects patterns in historical data and biases in what was studied, not a randomised causal effect.
</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (Unit D1)

### Q1. "Why do similar drugs tend to treat similar diseases (the guilt-by-association assumption)? When does it fail?"

**Why it tends to hold.**

1. **Structure → target.** Proteins recognise molecules through the shape and chemistry of a binding pocket. Molecules sharing substructures often fit the same pockets, so structurally similar drugs often share targets (the *similar property principle*; e.g. sildenafil and vardenafil, ECFP4 Tanimoto 0.59, both PDE5 inhibitors).
2. **Target → effect.** Drugs that modulate the same protein or pathway produce related physiological effects, so they help the diseases that depend on that pathway.
3. **Disease similarity → shared mechanism.** Diseases with similar symptoms, shared genes or nearby positions in an ontology often share pathophysiology, so a drug effective in one has a raised prior of working in another.
4. **Medical practice.** Drugs are developed and prescribed in classes; new class members are tested for the class's indications, which makes the pattern even stronger in indication databases.

Our data confirm it statistically: in Fdataset, two drugs share at least one indication 6.4 % of the time when their ECFP4 Tanimoto is below 0.2, but 82.8 % of the time when it is ≥ 0.8; MV-HGAT's propagation head is GBA written as matrix products.

**When it fails.**

1. *Activity cliffs*: tiny structural changes abolish or alter activity (≈ 30 % hit rate even at Tanimoto ≥ 0.85; Martin et al. 2002).
2. *Stereoisomers*: identical 2-D fingerprints, different pharmacology (quinine vs quinidine, Tanimoto 1.00, no shared indication).
3. *Same target, opposite effect by dosing*: pulsatile GnRH stimulates, continuous GnRH agonists suppress.
4. *Different scaffold, same target*: sildenafil vs tadalafil (0.115) — GBA misses true relationships (false negatives).
5. *Pharmacokinetics and route*: tissue exposure, BBB, topical vs systemic.
6. *Disease heterogeneity*: one disease label hides subtypes with different drivers; similar-looking diseases can have different causes.
7. *Biased or artefactual labels*: OMIM mapping oddities, historical indications, popularity bias of well-studied drugs and diseases.
8. *Agonist/antagonist blindness*: estrogens are "similar" to anti-estrogen-treated contexts but stimulate ER-positive breast cancer.
9. *Novel mechanisms*: GBA can only propose "more of the same"; first-in-class treatments are invisible to it, and cold-start entities have little to associate with.
10. *Circularity*: if similarity is derived from the labels, GBA just echoes them (leakage).

### Q2. "What would it take to turn a top-10 prediction into a real repurposed drug?"

A long sequence of increasingly expensive tests, most of which the prediction will fail (Section 3.14):

1. **Retrospective credibility** of the method: good performance on held-out known links under realistic protocols (warm and cold start, external datasets).
2. **Evidence triage, audited**: literature, curated databases, trial registries — reading what each hit actually is (our amantadine/memantine lesson).
3. **Mechanistic plausibility**: a biological rationale linking the drug's targets to the disease's pathophysiology; target expressed in the relevant tissue.
4. **Pharmacological feasibility**: can the drug reach the tissue at effective concentrations at a tolerable dose? Safety in the target population (age, kidney function, pregnancy), contraindications, interactions. A new dose or route may need new formulation and safety work.
5. **Preclinical experiments**: in-vitro assays in disease-relevant cells; in-vivo animal models with dose–response.
6. **Supporting real-world evidence** (optional but useful): health-record or claims analyses with careful confounding control.
7. **Phase II proof-of-concept RCT**: ethics approval, trial registration, a well-chosen endpoint, randomisation and blinding — the step where most candidates fail for lack of efficacy.
8. **Phase III confirmatory RCT(s)** with enough power.
9. **Regulatory approval** of the new indication: a supplemental NDA by the label holder or a 505(b)(2) application (US), or equivalent in the EU/India; possibly orphan designation for rare diseases.
10. **Post-approval**: guideline inclusion, reimbursement, Phase IV safety monitoring, uptake.
11. **Throughout: funding and incentives.** For an off-patent generic, someone (academia, a foundation, government, or a company using exclusivity or a new formulation) must pay for trials costing tens of millions of dollars. Repositioning typically takes years (estimates ≈ 6.5 years, ≈ US$300 million on average) even when it succeeds.

Until then, the top-10 entry is a **hypothesis**: useful for deciding which experiments to do next, never a basis for treating a patient.

---

## 9. Summary and cheat sheet

```
DRUG BASICS
  drug = substance intended to diagnose/treat/prevent disease; active moiety vs salt vs product
  small molecule (<~900 Da, synthetic, oral, SMILES, generics, NDA)
  biologic (protein, cells make it, injected, no SMILES, biosimilars, BLA)
  target = protein the drug binds (667 human proteins for 1,578 FDA drugs; Santos 2017)
  modes: agonist, antagonist, partial/inverse agonist, allosteric, enzyme inhibitor,
         channel blocker/opener, molecular glue/degrader (thalidomide -> cereblon)
  MoA = chain molecular event -> clinical effect (sildenafil: PDE5 -> cGMP -> vasodilation)

PD / PK
  Hill: E = Emax C^n / (EC50^n + C^n); potency (EC50) vs efficacy (Emax); therapeutic index
  ADME; F (bioavailability), Vd, CL;  k = CL/Vd;  t1/2 = 0.693 Vd / CL
  Css_avg = F*Dose/(CL*tau);  accumulation R = 1/(1-e^{-k tau});  ~94-97% of Css after 4-5 t1/2
  loading dose = C_target * Vd / F

LABELS
  indication (on label) | contraindication | off-label (legal, ~21% of US office prescriptions,
  73% of those weakly supported; Radley 2006) | polypharmacology (imatinib: BCR-ABL + KIT)

PIPELINE  discovery -> preclinical -> IND -> Phase I (20-100, safety) -> II (<= several hundred,
          efficacy signal, ~33% pass) -> III (300-3,000, confirm, 25-30% pass) -> NDA/BLA -> IV
  P(approval | Phase I) ~ 7-14% (3.4% oncology; Wong 2019); failures: 52% efficacy, 24% safety
  cost per approval US$1-2.6 bn; 10-15 years.  Kefauver-Harris 1962: proof of efficacy.

REPOSITIONING  new use for existing/failed drug; ~6.5 y and ~US$300 m (Nosengo 2016)
  saves safety/PK/manufacturing work; NOT efficacy risk; weak incentives for generics
  US routes: sNDA, 505(b)(2); 3-y new-use exclusivity; 7-y orphan exclusivity (<200k patients)

CASES  sildenafil (angina -> ED 1998 -> PAH 2005) | thalidomide (1957 sedative -> 1961 withdrawn ->
  ENL 1998 -> myeloma 2006) | minoxidil (1979 oral BP -> 1988 topical hair) | aspirin (1899 ->
  antiplatelet; CRC 2016 rec. withdrawn 2022) | metformin-cancer (2005 observational -> MA.32
  RCT 2022 negative) | baricitinib (RA -> COVID: KG 2020, EUA Nov 2020, approval May 2022)
  | AZT 1987 | propranolol-haemangioma 2008/2014 | imatinib CML 2001 -> GIST 2002
  | dexamethasone RECOVERY 2020 | hydroxychloroquine EUA revoked June 2020

COMPUTATIONAL  signature (CMap) | target/structure (SEA, docking) | network proximity |
  similarity ML (PREDICT, MBiRW, MF, GNNs = this project) | knowledge graphs (Hetionet, TxGNN) |
  real-world data | literature mining

GUILT BY ASSOCIATION  score = W A (drug side) or A W'^T (disease side) = MV-HGAT propagation head
  works on average (share-indication 6% -> 83% across Tanimoto bins); fails: activity cliffs,
  stereoisomers, dosing (GnRH), scaffold hopping (sildenafil/tadalafil 0.115), PK, heterogeneity,
  label artefacts, popularity bias, agonist/antagonist blindness, novelty, circularity

PREDICTION = HYPOTHESIS  audit evidence -> mechanism -> PK/safety -> in vitro -> in vivo -> RWD ->
  Phase II -> Phase III -> approval.  Never present a score as treatment advice.
```

---

## 10. Curated further resources

Links checked on 1 October 2026; paywalled journal pages verified through their DOIs. "Free" = readable without subscription.

**Reviews of drug repositioning**
- Pushpakom, S. et al. (2019). *Drug repurposing: progress, challenges and recommendations.* Nat Rev Drug Discov 18:41–58. [doi:10.1038/nrd.2018.168](https://doi.org/10.1038/nrd.2018.168) — the standard review: approaches, case studies, barriers. *Paid.*
- Ashburn, T. T. & Thor, K. B. (2004). *Drug repositioning: identifying and developing new uses for existing drugs.* Nat Rev Drug Discov 3:673–683. [doi:10.1038/nrd1468](https://doi.org/10.1038/nrd1468) — the classic that popularised the term; business case. *Paid.*
- Begley, C. G. et al. (2021). *Drug repurposing: misconceptions, challenges, and opportunities for academic researchers.* Sci Transl Med 13:eabd5524. [doi:10.1126/scitranslmed.abd5524](https://doi.org/10.1126/scitranslmed.abd5524) — a sober corrective on how hard repurposing really is. *Paid.*
- Nosengo, N. (2016). *Can you teach old drugs new tricks?* Nature 534:314–316. [doi:10.1038/534314a](https://doi.org/10.1038/534314a) — readable feature; source of the 6.5-year/US$300-million figures. *Paid (often accessible).*
- Jarada, T. N., Rokne, J. G. & Alhajj, R. (2020). *A review of computational drug repositioning: strategies, approaches, opportunities, challenges, and directions.* J Cheminform 12:46. [doi:10.1186/s13321-020-00450-7](https://doi.org/10.1186/s13321-020-00450-7) — broad survey of computational methods. *Free.*

**Pharmacology foundations**
- OpenStax, *Pharmacology for Nurses*. [link](https://openstax.org/details/books/pharmacology) — free textbook; clear chapters on PK/PD and drug classes. *Free.*
- *MSD Manual Professional Edition*, Clinical Pharmacology section. [link](https://www.merckmanuals.com/professional/clinical-pharmacology) — concise, authoritative pages on ADME, dose–response, drug interactions. *Free.*
- MIT OpenCourseWare, HST.151 *Principles of Pharmacology*. [link](https://ocw.mit.edu/courses/hst-151-principles-of-pharmacology-spring-2005/) — Harvard–MIT lecture notes on PK, PD and drug development. *Free.*
- Coursera / UC San Diego, *Drug Discovery*. [link](https://www.coursera.org/learn/drug-discovery) — an introductory course on the pipeline from target to clinic. *Free to audit.*
- Santos, R. et al. (2017). *A comprehensive map of molecular drug targets.* Nat Rev Drug Discov 16:19–34. [doi:10.1038/nrd.2016.230](https://doi.org/10.1038/nrd.2016.230) — which proteins drugs act on. *Paid.*
- Textbooks (paid): Rang & Dale's *Pharmacology*; Katzung's *Basic & Clinical Pharmacology*; Goodman & Gilman's *The Pharmacological Basis of Therapeutics* — reference works for any mechanism you meet.

**Drug development, regulation and economics**
- US FDA, *The Drug Development Process* (five steps, with the clinical-phase table). [link](https://www.fda.gov/patients/learn-about-drug-and-device-approvals/drug-development-process) — the authoritative plain-language overview. *Free.*
- US FDA, *Understanding Unapproved Use of Approved Drugs "Off Label"*. [link](https://www.fda.gov/patients/learn-about-expanded-access-and-other-treatment-options/understanding-unapproved-use-approved-drugs-label) — what off-label use is and isn't. *Free.*
- US FDA, *Applications Covered by Section 505(b)(2)* (draft guidance, 1999). [PDF](https://www.fda.gov/media/72419/download) — the regulatory route most relevant to repurposing. *Free.*
- Wong, C. H., Siah, K. W. & Lo, A. W. (2019). *Estimation of clinical trial success rates and related parameters.* Biostatistics 20:273–286. [doi:10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069) — modern success-rate estimates by phase and area. *Free (open access).*
- Harrison, R. K. (2016). *Phase II and phase III failures: 2013–2015.* Nat Rev Drug Discov 15:817–818. [doi:10.1038/nrd.2016.184](https://doi.org/10.1038/nrd.2016.184) — why late-stage trials fail. *Paid.*
- Wouters, O. J., McKee, M. & Luyten, J. (2020). *Estimated research and development investment needed to bring a new medicine to market, 2009–2018.* JAMA 323:844–853. [doi:10.1001/jama.2020.1166](https://doi.org/10.1001/jama.2020.1166) — transparent cost estimate from public data. *Free.*
- DiMasi, J. A., Grabowski, H. G. & Hansen, R. W. (2016). *Innovation in the pharmaceutical industry: new estimates of R&D costs.* J Health Econ 47:20–33. [doi:10.1016/j.jhealeco.2016.01.012](https://doi.org/10.1016/j.jhealeco.2016.01.012) — the US$2.6-billion estimate. *Paid.*
- Radley, D. C., Finkelstein, S. N. & Stafford, R. S. (2006). *Off-label prescribing among office-based physicians.* Arch Intern Med 166:1021–1026. [doi:10.1001/archinte.166.9.1021](https://doi.org/10.1001/archinte.166.9.1021) — how common off-label use is. *Paid.*

**Case histories (primary and review sources)**
- Ghofrani, H. A., Osterloh, I. H. & Grimminger, F. (2006). *Sildenafil: from angina to erectile dysfunction to pulmonary hypertension and beyond.* Nat Rev Drug Discov 5:689–702. [free full text (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7097805/) · [doi:10.1038/nrd2030](https://doi.org/10.1038/nrd2030). *Free via PMC.*
- Vargesson, N. (2015). *Thalidomide-induced teratogenesis: history and mechanisms.* Birth Defects Res C 105:140–156. [doi:10.1002/bdrc.21096](https://doi.org/10.1002/bdrc.21096) — history and biology of the tragedy. *Free (CC-BY).*
- Ito, T. et al. (2010). *Identification of a primary target of thalidomide teratogenicity.* Science 327:1345–1350. [doi:10.1126/science.1177319](https://doi.org/10.1126/science.1177319) — cereblon. *Paid.*
- Donovan, K. A. et al. (2018). *Thalidomide promotes degradation of SALL4…* eLife 7:e38430. [doi:10.7554/eLife.38430](https://doi.org/10.7554/eLife.38430) — the molecular-glue mechanism of birth defects. *Free.*
- Sheskin, J. (1965). *Thalidomide in the treatment of lepra reactions.* Clin Pharmacol Ther 6:303–306. [doi:10.1002/cpt196563303](https://doi.org/10.1002/cpt196563303). *Paid.*
- Rothwell, P. M. et al. (2010). *Long-term effect of aspirin on colorectal cancer incidence and mortality.* Lancet 376:1741–1750. [doi:10.1016/S0140-6736(10)61543-7](https://doi.org/10.1016/S0140-6736(10)61543-7). *Paid.*
- US Preventive Services Task Force (2022). *Aspirin use to prevent cardiovascular disease: preventive medication.* [link](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-preventive-medication) — why the colorectal-cancer recommendation was withdrawn. *Free.*
- Evans, J. M. M. et al. (2005). *Metformin and reduced risk of cancer in diabetic patients.* BMJ 330:1304–1305. [doi:10.1136/bmj.38415.708634.F7](https://doi.org/10.1136/bmj.38415.708634.F7). *Free.*
- Goodwin, P. J. et al. (2022). *Effect of metformin vs placebo on invasive disease–free survival in patients with breast cancer: the MA.32 randomized clinical trial.* JAMA 327:1963–1973. [doi:10.1001/jama.2022.6147](https://doi.org/10.1001/jama.2022.6147). *Paid.*
- Richardson, P. et al. (2020). *Baricitinib as potential treatment for 2019-nCoV acute respiratory disease.* Lancet 395:e30–e31. [doi:10.1016/S0140-6736(20)30304-4](https://doi.org/10.1016/S0140-6736(20)30304-4) — the knowledge-graph prediction. *Free.*
- Marconi, V. C. et al. (2021). *Efficacy and safety of baricitinib for the treatment of hospitalised adults with COVID-19 (COV-BARRIER).* Lancet Respir Med 9:1407–1418. [doi:10.1016/S2213-2600(21)00331-3](https://doi.org/10.1016/S2213-2600(21)00331-3). *Free to read (COVID-19 collection).*
- Kalil, A. C. et al. (2021). *Baricitinib plus remdesivir for hospitalized adults with Covid-19 (ACTT-2).* NEJM 384:795–807. [doi:10.1056/NEJMoa2031994](https://doi.org/10.1056/NEJMoa2031994). *Free.*
- RECOVERY Collaborative Group (2021). *Dexamethasone in hospitalized patients with Covid-19.* NEJM 384:693–704. [doi:10.1056/NEJMoa2021436](https://doi.org/10.1056/NEJMoa2021436). *Free.*
- Léauté-Labrèze, C. et al. (2008). *Propranolol for severe hemangiomas of infancy.* NEJM 358:2649–2651. [doi:10.1056/NEJMc0708819](https://doi.org/10.1056/NEJMc0708819). *Paid.*

**Computational repositioning**
- Gottlieb, A. et al. (2011). *PREDICT: a method for inferring novel drug indications with application to personalized medicine.* Mol Syst Biol 7:496. [doi:10.1038/msb.2011.26](https://doi.org/10.1038/msb.2011.26) — origin of Fdataset; read the Introduction. *Free.*
- Luo, H. et al. (2016). *Drug repositioning based on comprehensive similarity measures and Bi-Random walk algorithm.* Bioinformatics 32:2664–2671. [doi:10.1093/bioinformatics/btw228](https://doi.org/10.1093/bioinformatics/btw228) — origin of Cdataset and the MBiRW baseline. *Free.*
- Lamb, J. et al. (2006). *The Connectivity Map.* Science 313:1929–1935. [doi:10.1126/science.1132939](https://doi.org/10.1126/science.1132939) — signature-based repositioning. *Paid.*
- Sirota, M. et al. (2011). *Discovery and preclinical validation of drug indications using compendia of public gene expression data.* Sci Transl Med 3:96ra77. [doi:10.1126/scitranslmed.3001318](https://doi.org/10.1126/scitranslmed.3001318); replication: Kandela, I. et al. (2017) eLife 6:e17044, [doi:10.7554/eLife.17044](https://doi.org/10.7554/eLife.17044) (*free*) — a prediction, its validation, and its replication.
- Keiser, M. J. et al. (2009). *Predicting new molecular targets for known drugs.* Nature 462:175–181. [doi:10.1038/nature08506](https://doi.org/10.1038/nature08506) — ligand-similarity target prediction (SEA). *Paid.*
- Guney, E. et al. (2016). *Network-based in silico drug efficacy screening.* Nat Commun 7:10331. [doi:10.1038/ncomms10331](https://doi.org/10.1038/ncomms10331) — network proximity. *Free.*
- Cheng, F. et al. (2018). *Network-based approach to prediction and population-based validation of in silico drug repurposing.* Nat Commun 9:2691. [doi:10.1038/s41467-018-05116-5](https://doi.org/10.1038/s41467-018-05116-5) — prediction plus patient-data validation. *Free.*
- Himmelstein, D. S. et al. (2017). *Systematic integration of biomedical knowledge prioritizes drugs for repurposing.* eLife 6:e26726. [doi:10.7554/eLife.26726](https://doi.org/10.7554/eLife.26726) — Hetionet / Project Rephetio. *Free.*
- Huang, K. et al. (2024). *A foundation model for clinician-centered drug repurposing.* Nature Medicine 30:3601–3613. [doi:10.1038/s41591-024-03233-x](https://doi.org/10.1038/s41591-024-03233-x) — TxGNN, zero-shot repurposing. *Free.*
- Corsello, S. M. et al. (2017). *The Drug Repurposing Hub: a next-generation drug library and information resource.* Nat Med 23:405–408. [doi:10.1038/nm.4306](https://doi.org/10.1038/nm.4306) · [the Hub](https://repo-hub.broadinstitute.org/repurposing) — curated compounds with targets and clinical phases. *Hub free.*

**Guilt by association and its limits**
- Martin, Y. C., Kofron, J. L. & Traphagen, L. M. (2002). *Do structurally similar molecules have similar biological activity?* J Med Chem 45:4350–4358. [doi:10.1021/jm020155c](https://doi.org/10.1021/jm020155c). *Paid.*
- Gillis, J. & Pavlidis, P. (2012). *"Guilt by association" is the exception rather than the rule in gene networks.* PLoS Comput Biol 8:e1002444. [doi:10.1371/journal.pcbi.1002444](https://doi.org/10.1371/journal.pcbi.1002444). *Free.*

**Data tools**
- ClinicalTrials.gov API documentation. [link](https://clinicaltrials.gov/data-api/api) — the API used by `06_case_study.py`; read how search terms are expanded. *Free.*

---

## 11. Glossary

- **Absorption / bioavailability (F)** — entry of drug into the blood; fraction of a dose reaching circulation unchanged.
- **Active moiety** — the part of a molecule responsible for its action (without salt partners).
- **ADME** — absorption, distribution, metabolism, excretion.
- **Agonist / antagonist** — drug that activates / blocks a receptor.
- **Activity cliff** — a small structural change causing a large change in activity.
- **Biologic** — a drug made by living cells, typically a protein (e.g. antibody); approved via BLA.
- **Biosimilar** — a highly similar copy of a biologic after patent expiry.
- **Blood–brain barrier (BBB)** — the barrier restricting entry of molecules into the brain.
- **Clearance (CL)** — volume of plasma cleared of drug per unit time.
- **Confounding** — a third factor creating a spurious association between treatment and outcome.
- **Contraindication** — a situation in which a drug should not be used.
- **CYP450** — liver enzymes that metabolise many drugs.
- **Drug repositioning / repurposing** — finding new uses for existing or failed drugs.
- **EC<sub>50</sub> / E<sub>max</sub>** — concentration for half-maximal effect (potency) / maximal effect (efficacy).
- **Emergency Use Authorization (EUA)** — temporary FDA authorisation during emergencies.
- **Enantiomers / stereoisomers** — molecules with the same connectivity but different 3-D arrangement.
- **Generic** — an identical copy of a small-molecule drug after patent expiry.
- **Guilt by association (GBA)** — inferring properties from similar entities.
- **Half-life (t<sub>1/2</sub>)** — time for the concentration to fall by half.
- **Indication** — an approved use stated on a drug's label.
- **IND** — Investigational New Drug application, needed to start US human trials.
- **Kefauver–Harris Amendments (1962)** — US law requiring proof of efficacy.
- **Knowledge graph** — a graph integrating many entity and relation types.
- **Label (prescribing information)** — the regulator-approved document describing uses, doses and risks.
- **Mechanism of action (MoA)** — how a drug's molecular action produces its clinical effect.
- **Molecular glue / degrader** — a drug that induces degradation of specific proteins (e.g. via cereblon).
- **NDA / sNDA / 505(b)(2)** — US New Drug Application; supplement for new indications; route allowing reliance on others' data.
- **Network proximity** — closeness of drug targets to disease genes in the interactome.
- **Off-label use** — prescribing an approved drug for an unapproved indication, dose or population.
- **Orphan drug** — a drug for a rare disease (US: < 200,000 patients), with 7-year exclusivity.
- **Pharmacodynamics (PD)** — what a drug does to the body.
- **Pharmacokinetics (PK)** — what the body does to a drug.
- **Phase I / II / III / IV** — clinical trial stages: safety/dose; efficacy signal; confirmation; post-marketing.
- **Polypharmacology** — one drug acting on several targets.
- **Prodrug** — an inactive precursor converted to the active drug in the body.
- **Randomised controlled trial (RCT)** — trial with random allocation to treatment or control.
- **Signature-based repositioning** — matching drug and disease gene-expression signatures.
- **Similar property principle** — similar molecules tend to have similar properties.
- **Small molecule** — a low-molecular-weight, chemically synthesised drug.
- **Target** — the biological molecule (usually a protein) a drug binds to produce its effect.
- **Therapeutic index / window** — margin between effective and toxic exposure.
- **Volume of distribution (V<sub>d</sub>)** — apparent volume relating amount in body to plasma concentration.
