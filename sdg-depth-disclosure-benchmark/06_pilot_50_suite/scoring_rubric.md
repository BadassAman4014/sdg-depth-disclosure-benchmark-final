# 📘 SDG Depth Scoring Rubric (0 to 5 Scale)
### Reference: *Izhar et al. (2026)*, Section 3.4.3 & Table 1; *Hummel (2019)*; *PwC (2018)*

This rubric defines the scoring criteria for evaluating the depth of corporate SDG disclosures on a scale from **0 to 5**.

---

## 📊 Table 1: Official Scoring Matrix (*Izhar et al., 2026*)

| Score | Description / Theoretical Construct | Qualitative Target | Quantitative Target | SDG-Related Actions | Qualitative Measurement of Outcome | Quantitative Measurement of Outcome |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | **No SDG Information / Pure Boilerplate** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **1** | **Qualitative Target Only** (Aspiration / Vision) | ✔️ | ❌ | ❌ | ❌ | ❌ |
| **2** | **Qualitative Target + SDG Actions** OR **Quantitative Target** | ✔️ <br> *(or ❌)* | *(or ✔️)* <br> ❌ | ✔️ <br> *(or ❌)* | ❌ | ❌ |
| **3** | **Qualitative Target + Qualitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ✔️ | ❌ |
| **4** | **Qualitative Target + Quantitative Outcome** | ✔️ | ❌ | *(often ✔️)* | ❌ | ✔️ |
| **5** | **Quantitative Target + Measured Outcome** (Qual or Quant) | ❌ | ✔️ | *(often ✔️)* | ✔️ | ✔️ |

---

## 🔍 Linguistic Component Definitions:

1. **Qualitative Target (`qual_target`):**
   * High-level aspirations, pledges, or commitments without a quantified KPI.
   * *Keywords:* *"aim to"*, *"strive to"*, *"aspire"*, *"pledge"*, *"committed to promoting"*, *"seek to foster"*, *"vision"*.
2. **Quantitative Target (`quant_target`):**
   * Specific numerical targets or deadlines.
   * *Keywords:* *"reduce emissions by 30% by 2030"*, *"target zero waste by 2025"*, *"reach 100% renewable electricity"*.
3. **SDG-Related Actions (`action`):**
   * Concrete operational activities, infrastructure investments, training, or equipment installation.
   * *Keywords:* *"installed"*, *"implemented"*, *"commissioned"*, *"invested €5M"*, *"constructed"*, *"trained 1,200 employees"*, *"deployed"*.
4. **Qualitative Measurement of Outcome (`qual_outcome`):**
   * Narrative description of results, progress, or positive impact achieved.
   * *Keywords:* *"led to improved workplace safety"*, *"resulted in enhanced biodiversity"*, *"strengthened supplier resilience"*.
5. **Quantitative Measurement of Outcome (`quant_outcome`):**
   * Hard verifiable metrics of past progress or third-party audit.
   * *Keywords:* *"reduced Scope 1 emissions by 24.5%"*, *"saved 1.2 million kWh"*, *"diverted 85% of waste from landfills"*, *"verified by PwC/TÜV"*.

---

## 🔀 Decision Flowchart for Coders:

```
Start with Passage
  │
  ├── Does it mention any corporate SDG target, action, or outcome?
  │     ├── NO  ──> SCORE 0 (Boilerplate / Index header / Macro report)
  │     └── YES ──> Continue below
  │
  ├── Does it have a QUANTITATIVE TARGET?
  │     ├── YES:
  │     │     ├── Does it report a measured outcome (Qual or Quant)?
  │     │     │     ├── YES ──> SCORE 5 (Maximum depth)
  │     │     │     └── NO  ──> SCORE 2 (Quantitative target alone)
  │     └── NO:
  │           ├── Does it report a QUANTITATIVE MEASUREMENT OF OUTCOME?
  │           │     ├── YES ──> SCORE 4 (Qual target + Quant outcome)
  │           │     └── NO:
  │           │           ├── Does it report a QUALITATIVE MEASUREMENT OF OUTCOME?
  │           │           │     ├── YES ──> SCORE 3 (Qual target + Qual outcome)
  │           │           │     └── NO:
  │           │                 ├── Does it describe concrete IMPLEMENTED ACTIONS?
  │           │                 │     ├── YES ──> SCORE 2 (Qual target + Action)
  │           │                 │     └── NO  ──> SCORE 1 (Qual target only / Aspiration)
```

---

## 🔄 Mapping to Binary Classification (`sym` vs. `sub`):
* **Scores 0 and 1** $\longrightarrow$ **Symbolic (`sym`)**: Aspirational or general policy claims without verifiable operational outcomes.
* **Scores 2, 3, 4, and 5** $\longrightarrow$ **Substantive (`sub`)**: Tangible actions, targets, and measured results.
