# 3D Deep Learning — Medical AI Image Segmentation

**Role:** ML/AI Engineer — Image segmentation with 3D deep learning and graph-based feature modeling.  
**When:** Jan. 2025 – Jun. 2026 · **Location:** Taipei, Taiwan

> **Disclaimer:** This portfolio presents sanitized project descriptions, aggregated research results, and publication-safe examples.  
> Raw clinical data, identifiable patient information, restricted code, and confidential research materials are not included.

---

## Project at a Glance

| Workstream | Problem | What I Built | Outcome | Key Skills |
|---|---|---|---|---|
| **1. Problem Definition** | Coronary OCT calcification is difficult to segment because of low contrast, signal attenuation, artifacts, and highly variable lesion morphology | Framed the task as a **3D medical image segmentation problem** and identified key failure modes that should guide model design | Defined clear technical objectives for robustness, spatial-context modeling, and boundary delineation | Problem Framing, Medical Imaging, Segmentation |
| **2. Data Preparation** | Clinical OCT data were large-scale, heterogeneous, and required patient-level organization before training | Structured **138 patients, 192 pullbacks, and 34,954 frames** into a consistent dataset with controlled train/validation/test organization | Created a reproducible dataset foundation for model development and fair comparison | Data Processing, Dataset Design, Data Validation |
| **3. Preprocessing Pipeline** | Raw 2D OCT frames were not directly suitable for volumetric learning | Built an end-to-end pipeline for **polar transformation, artifact handling, signal enhancement, augmentation, and 3D volume construction** | Converted **704×704 frames into 256×256×8 model-ready volumes** with longitudinal context | Python, Image Processing, Data Pipeline, 3D Data |
| **4. Model Design** | Standard 3D CNNs mainly capture local context and may underrepresent long-range structural relationships | Developed and compared **3D U-Net, attention, GCN, graph structural filtering, gated fusion, and non-local** architectures | Identified **GSF_GAP** as the strongest overall model with mean **3D Dice 0.834** | Deep Learning, 3D CNN, GNN, Architecture Design |
| **5. Evaluation & Results** | Architecture improvements needed to be validated across repeated experiments rather than a single split | Designed repeated train/validation/test experiments with **Dice, Precision, Recall, IoU, paired t-tests, and Friedman testing** | Achieved best mean **Dice 0.834** and **IoU 0.703**, while quantifying precision–recall trade-offs across models | Experiment Design, Statistical Testing, Model Evaluation |

---

# Project Details

## P1 — Problem Definition

**Problem**  
Coronary calcification is difficult to segment in OCT because lesion boundaries can be weak or incomplete, signal intensity decreases with tissue depth, imaging artifacts obscure local anatomy, and calcified plaques vary substantially in thickness, arc, and morphology.

**Solution**  
Framed the task as a **3D medical image segmentation problem** so the model could use both cross-sectional appearance and information from adjacent OCT frames. The problem definition was translated into three technical objectives: improve robustness to low-contrast regions, capture broader spatial context, and preserve difficult lesion boundaries.

**Technical Highlights**
- Identified major failure modes including **low contrast, signal attenuation, guidewire artifacts, and heterogeneous lesion morphology**.
- Defined a **3D volumetric formulation** rather than treating each OCT frame independently.
- Used the observed failure modes to guide later choices in preprocessing, graph-based contextual modeling, and evaluation.

**Impact**
- Established a clear set of engineering requirements that connected the clinical imaging challenge to the downstream **data pipeline, model architecture, and evaluation strategy**.

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Coronary OCT"] --> B["Low Contrast / Signal Attenuation"]
    A --> C["Artifacts"]
    A --> D["Variable Lesion Morphology"]
    B --> E["Segmentation Challenges"]
    C --> E
    D --> E
    E --> F["3D Context + Robust Boundary Modeling"]
    F --> G["Model & Evaluation Requirements"]
```

[Deep Dive →](docs/01_problem.md)

---

## P2 — Data

**Problem**  
_TODO_

**Solution**  
_TODO_

**Technical Highlights**
- _TODO_
- _TODO_
- _TODO_

**Impact**
- _TODO_

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Clinical Pullbacks"] --> B["Patient / Pullback Organization"]
    B --> C["Frame Extraction"]
    C --> D["Annotation Alignment"]
    D --> E["Dataset Split"]
    E --> F["Model Dataset"]
```

[Deep Dive →](docs/02_data.md)

---

## P3 — Pipeline

**Problem**  
_TODO_

**Solution**  
_TODO_

**Technical Highlights**
- _TODO_
- _TODO_
- _TODO_

**Impact**
- _TODO_

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Raw OCT Frames"] --> B["Preprocessing"]
    B --> C["Polar Transformation"]
    C --> D["Artifact / Signal Handling"]
    D --> E["Augmentation"]
    E --> F["3D Volume Construction"]
    F --> G["Model-ready Input"]
```

[Deep Dive →](docs/03_pipeline.md)

---

## P4 — Model Design

**Problem**  
_TODO_

**Solution**  
_TODO_

**Technical Highlights**
- _TODO_
- _TODO_
- _TODO_

**Impact**
- _TODO_

**Architecture / Workflow**

```mermaid
flowchart LR
    A["3D U-Net Baseline"] --> B["Attention"]
    B --> C["Graph Modeling"]
    C --> D["Graph Structural Filtering"]
    D --> E["Gated Fusion"]
    A --> F["Non-local Baseline"]
```

[Deep Dive →](docs/04_model_design.md)

---

## P5 — Experiment & Results

**Problem**  
_TODO_

**Solution**  
_TODO_

**Technical Highlights**
- _TODO_
- _TODO_
- _TODO_

**Impact**
- _TODO_

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Train"] --> B["Validate"]
    B --> C["Test"]
    C --> D["Repeat"]
    D --> E["Aggregate Metrics"]
    E --> F["Statistical Testing"]
    F --> G["Failure Analysis"]
```

[Deep Dive →](docs/05_experiments_results.md)

---

## Key Engineering Takeaways

_TODO_

---

## Tech Stack

| Area | Technologies |
|---|---|
| Programming | _TODO_ |
| Deep Learning | _TODO_ |
| Graph Learning | _TODO_ |
| Image Processing | _TODO_ |
| Experimentation | _TODO_ |
| Visualization | _TODO_ |
| Compute / Engineering | _TODO_ |

---

## Context / Disclaimer

**Project Context**  
_TODO_

**Data Privacy**  
_TODO_

**Repository Scope**  
_TODO_
