# 3D Deep Learning — Medical AI Image Segmentation

**Role:** ML/AI Engineer — Image segmentation with 3D deep learning and graph-based feature modeling.  
**When:** Jan. 2025 – Jun. 2026 · **Location:** Taipei, Taiwan

> **Disclaimer:** This portfolio presents sanitized project descriptions, aggregated research results, and publication-safe examples.  
> Raw clinical data, identifiable patient information, restricted code, and confidential research materials are not included.

---

## Project at a Glance

| Workstream | Problem | What I Built | Outcome | Key Skills |
|---|---|---|---|---|
| **1. Problem Definition** | Coronary OCT calcification is difficult to segment because of weak signals, artifacts, limited 2D context, and complex lesion boundaries | Translated imaging failure modes into clear requirements for **robustness, wider spatial context, and boundary quality** | Established the technical direction for the full ML pipeline | Problem Framing, Medical Imaging, Segmentation |
| **2. Data Preparation** | Raw clinical OCT scans, metadata, and physician annotations were not directly usable for model training | Built a structured workflow to ingest **34,954 frames**, convert clinical annotations into pixel-level masks, and preserve pullback sequence information | Created a standardized segmentation dataset linking images, labels, metadata, and frame order | Data Engineering, DICOM, Annotation Processing, Dataset Design |
| **3. Preprocessing Pipeline** | Standardized OCT frames still contained imaging noise, artifacts, signal loss, and limited longitudinal context | Built a domain-specific pipeline for **lumen noise removal, polar transformation, guidewire-shadow handling, signal enhancement, augmentation, and 3D VoI construction** | Transformed **704×704 OCT frames into standardized 256×256×8 3D inputs with longitudinal context** | Image Processing, MATLAB, Data Augmentation, 3D Data Pipeline |
| **4. Model Design** | Standard 3D CNNs mainly capture local patterns and may miss wider structural relationships | Built a shared **3D Attention U-Net backbone**, then evaluated **graph message passing, Graph Structural Filtering, and alternative global-context baselines** | **GSF_GAP** achieved the highest mean **3D Dice of 0.834** | Deep Learning, 3D CNN, GNN, Architecture Design |
| **5. Model Validation & Results** | Model improvements needed to be validated at the vessel level across repeated experiments rather than individual VoIs or one favorable split | Aggregated overlapping VoI predictions into vessel-level outputs and evaluated models using **Dice, Precision, Recall, IoU, and paired analysis** | Achieved best mean **Dice 0.834** and **IoU 0.703**, while identifying precision–recall trade-offs across architectures | Experiment Design, Statistical Testing, Model Evaluation |

---

## End-to-End System Overview

```mermaid
flowchart TB
    A["Raw Imaging Data +<br/>Ground-Truth Labels"]
    B["P2 · Data Preparation<br/>DICOM · Masks · Sequence Mapping"]
    C["P3 · Preprocessing<br/>Polar Transform · Artifact Handling · Signal Enhancement"]
    D["3D Inputs<br/>256×256×8"]
    E["P4 · Shared 3D Attention U-Net Backbone"]

    U["U-Net<br/>Image Baseline"]
    G["GCN U-Net<br/>Graph Baseline"]
    S["GSF_GAP U-Net<br/>Main Method"]
    V["GSF Variants<br/>RMSP · Soft Gating"]
    X["Comparison Models<br/>CWA · Non-local"]

    F["P5 · Vessel-level Evaluation<br/>3 Disjoint Test Sets · 134 Vessels"]
    R["Dice · Precision · Recall · IoU"]

    A --> B --> C --> D --> E
    E --> U
    E --> G --> S
    S --> V
    E -. "alternative strategies" .-> X

    U --> F
    G --> F
    S --> F
    V --> F
    X --> F
    F --> R
```

**System logic:** clinical data are converted into standardized 3D inputs, processed through a controlled model family, and compared under the same vessel-level validation framework. The main research path progresses from **3D CNN features → graph message passing → Graph Structural Filtering**.

---

# Project Details

## P1 — Problem Definition

**Problem**  
Coronary calcification is difficult to segment in OCT because lesion boundaries can be unclear, deeper tissue has weaker signals, imaging artifacts can hide local structures, and calcified plaques vary greatly in shape and size.

**Solution**  
Defined the task as a **3D medical image segmentation problem** so the model could use both each OCT frame and nearby frames. This led to three main design goals: handle weak-signal regions better, capture wider spatial context, and improve boundary segmentation.

**Technical Highlights**
- **Translated real OCT failure modes into concrete model requirements:** robustness to weak signals and artifacts, stronger spatial-context modeling, and accurate boundary segmentation.
- **Moved beyond frame-by-frame segmentation by designing a 3D input setup**, allowing the model to learn lesion continuity across neighboring OCT slices.
- **Built the technical roadmap around these constraints**, connecting preprocessing, 3D volume construction, graph-based feature modeling, and evaluation into one end-to-end segmentation workflow.

**Impact**
- **Established a clear technical direction for the full ML pipeline**, turning clinical imaging challenges into specific decisions for preprocessing, 3D input design, graph-based modeling, and evaluation.

[Deep Dive →](docs/01_problem.md)

---

## P2 — Data Preparation

**Problem Snapshot**

| Challenge | Why It Matters |
|---|---|
| **Raw clinical data** | 34,954 OCT frames came from 138 patients and 192 pullbacks, with separate image, metadata, and annotation sources |
| **Clinical annotations** | Physician-defined coordinates were not directly usable as segmentation masks |
| **Sequential structure** | Frame order within each pullback had to be preserved for later 3D modeling |

**Engineering Response**

| Decision | What I Built |
|---|---|
| **Structured ingestion** | Read OCT DICOM images and metadata and linked them by patient, pullback, and frame |
| **Label construction** | Converted two-physician consensus annotations into pixel-level segmentation masks |
| **Sequence preservation** | Maintained frame order and image-mask alignment to create standardized OCT sequences |

**Technical Highlights**
- Processed **34,954 OCT frames from 138 patients and 192 pullbacks**, including **26,648 calcified** and **8,306 non-calcified frames**.
- Preserved **frame order and pullback continuity** using OCT metadata, keeping the longitudinal structure required for later 3D modeling.
- Standardized image-mask mapping and geometry across the dataset to create a consistent interface for the preprocessing pipeline.

**Impact**
- **Established a structured, model-ready OCT dataset**, providing a reliable foundation for downstream preprocessing and 3D model development.

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Raw OCT DICOM"] --> C["DICOM + Metadata Ingestion"]
    B["Clinical Annotations"] --> D["Annotation Parsing"]

    C --> E["Frame & Pullback Mapping"]
    D --> F["Pixel-level Mask Construction"]

    E --> G["Image–Label Alignment"]
    F --> G

    G --> H["Ordered OCT Sequences"]
    H --> I["P3: Preprocessing"]
```

[Deep Dive →](docs/02_data.md)

---

## P3 — Preprocessing Pipeline

**Problem Snapshot**

| Challenge | Why It Matters |
|---|---|
| **Imaging noise & artifacts** | Lumen-region noise and guidewire shadows could introduce misleading features |
| **Weak deep-tissue signals** | Calcification farther from the lumen can have lower contrast and unclear boundaries |
| **OCT geometry** | Raw Cartesian images do not naturally represent the radial vessel-wall structure |
| **Limited 2D context** | A single frame cannot capture lesion continuity along the vessel |

**Engineering Response**

| Decision | What I Built |
|---|---|
| **Clean the signal** | Removed lumen-region noise before downstream processing |
| **Normalize & vary the geometry** | Converted Cartesian OCT frames to a 256×256 polar representation and generated center-shift variations during transformation |
| **Refine tissue features** | Removed guidewire shadows and enhanced weak deep-tissue signals |
| **Build 3D context** | Constructed overlapping 8-slice 3D VoIs with stride 3 |

**Technical Highlights**
- Used **U-Net-based lumen segmentation** to remove lumen-region noise before downstream image processing.
- Transformed OCT frames from **Cartesian to 256×256 polar coordinates** and introduced **center-shift augmentation** through alternative transformation centers.
- Combined **guidewire-shadow removal and depth-aware signal enhancement** to reduce artifacts and strengthen weak deep-tissue features.
- Constructed overlapping **8-slice 3D VoIs with stride 3**, producing **256×256×8×1 inputs** that preserve longitudinal lesion context.

**Impact**
- **Built a robust 3D input pipeline for volumetric segmentation**, reducing OCT-specific noise while preserving vessel structure and longitudinal lesion context.

**Architecture / Workflow**

```mermaid
flowchart LR
    A["Standardized OCT Sequences"]

    subgraph S1["1. Clean"]
        B["Lumen Segmentation"]
        C["Noise Removal + Grayscale"]
    end

    subgraph S2["2. Normalize"]
        D["Cartesian → Polar"]
        E["Resize to 256×256"]
        F["Center-shift Augmentation"]
    end

    subgraph S3["3. Refine"]
        G["Guidewire Shadow Removal"]
        H["Depth-aware Enhancement"]
    end

    subgraph S4["4. Build 3D Context"]
        I["8-slice Sliding Window"]
        J["Stride = 3"]
    end

    K["256×256×8×1 Model Input"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> G
    D -. "training augmentation" .-> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

[Deep Dive →](docs/03_pipeline.md)

---

## P4 — Model Design

> **Design Goal:** Extend a shared 3D segmentation backbone with explicit graph-based long-range modeling, then refine graph features before message passing through Graph Structural Filtering.

### Architecture Evolution

> **Design progression:** 3D feature extraction → graph-based long-range modeling → Graph Structural Filtering.

```mermaid
flowchart LR
    A["3D Attention U-Net<br/>Image Baseline"]

    A -->|"Need explicit<br/>long-range relations"| B["GCN U-Net<br/>Graph Baseline"]

    B -->|"Can graph features be<br/>refined before propagation?"| C["GSF_GAP U-Net<br/>Main Method"]

    C --> D["GSF_RMSP<br/>Frequency Descriptor Variant"]
    C --> E["Gated GSF_GAP<br/>Soft-filtering Variant"]

    A -. "Alternative feature modeling" .-> F["CWA U-Net"]
    A -. "Alternative global modeling" .-> G["Non-local U-Net"]
```

---

### Key Design Decisions

| Area | Decision | Engineering Rationale |
|---|---|---|
| **3D Backbone** | 8 slices · 32 base channels · GroupNorm | Improve cross-slice context and stabilize small-batch training |
| **Graph Representation** | 128 nodes · KNN `k=4` | Reduce graph computation while preserving the 8-slice depth structure |
| **Message Passing** | 2× GraphConv + residual connections | Aggregate relational context while preserving node features |
| **Structural Filtering** | GFT + Frequency SE | Adapt graph-frequency contributions before propagation |

### Technical Highlights

- **Designed a 3D Attention U-Net backbone** with multi-scale and cross-slice feature extraction using ASPP, depth-dilated Conv3D, and attention-gated skip connections.
- **Built a compact graph representation** from multi-scale CNN features using 128 nodes, KNN connectivity, and residual GraphConv for long-range dependency modeling.
- **Developed Graph Structural Filtering (GSF)** with Graph Fourier Transform and Frequency SE to adaptively reweight graph-frequency components before message passing.

**Validation design:** GAP vs. RMSP vs. soft gating, with CWA and Non-local attention as alternative modeling baselines.

### Architecture Pipeline

```mermaid
flowchart LR
    A["3D OCT Input<br/>256×256×8×1"]
    B["① Extract 3D Features<br/>3D Attention U-Net"]
    C["② Build Relational Graph<br/>128 nodes · KNN k=4"]
    D["③ Apply Structural Filtering<br/>GFT · Frequency SE · Reweighting"]
    E["④ Propagate Long-range Context<br/>2× GraphConv"]
    F["⑤ Decode Segmentation<br/>3D Decoder"]

    A --> B
    B -->|"multi-scale features"| C
    C -->|"graph features"| D
    D -->|"filtered features"| E
    E -->|"context-aware features"| F
```

**How it works:** The backbone first extracts **3D spatial and cross-slice features**. These features are compressed into a **128-node KNN graph** to model long-range relationships. GSF then **reweights graph-frequency information before message passing**, and the refined graph features are decoded into the final segmentation mask.

### Impact

- Established a controlled model design that separates the effects of the **3D backbone, graph message passing, and Graph Structural Filtering**.
- **GSF_GAP U-Net achieved the highest mean 3D Dice of 0.834**, higher than the U-Net and GCN baselines under the same evaluation setup.

[Model Design Deep Dive →](docs/04_model_design.md)

---

## P5 — Model Validation & Results

> **Validation Goal:** Determine whether graph modeling and Graph Structural Filtering add measurable value beyond the shared 3D segmentation backbone.

### Validation Framework

| Design | Implementation | Why It Matters |
|---|---|---|
| **Controlled comparison** | Same preprocessing, training setup, and data splits | Isolate architecture changes |
| **Repeated testing** | 3 disjoint test sets · 134 vessels | Reduce dependence on one split |
| **Vessel-level evaluation** | Aggregate overlapping VoI predictions before scoring | Evaluate the final segmentation output |
| **Multi-metric analysis** | Dice · Precision · Recall · IoU | Capture overlap and prediction trade-offs |

### Architecture Evidence

| Architecture Step | Mean 3D Dice | Change |
|---|---:|---:|
| **3D U-Net** | 0.818 | Baseline |
| **+ Graph Modeling** | 0.823 | +0.005 |
| **+ Graph Structural Filtering** | **0.834** | **+0.011 vs. GCN** |

**Key finding:** The largest architecture-stage gain appeared after adding **Graph Structural Filtering before message passing**.

### Key Findings

- **GSF_GAP** achieved the highest mean 3D Dice: **0.834 ± 0.033**.
- **GSF_RMSP** achieved the highest Precision: **0.824 ± 0.032**, reflecting more conservative predictions.
- **U-Net** achieved the highest Recall: **0.861 ± 0.029**, reflecting broader lesion coverage.
- Repeated testing showed that single-split rankings could change, so model selection was based on **cross-test behavior rather than one favorable result**.
- Paired comparisons were treated as **exploratory evidence** because multiple-comparison correction was not applied.

[Model Validation & Results Deep Dive →](docs/05_experiments_results.md)

---

## Key Engineering Takeaways

- **Built an end-to-end 3D medical imaging pipeline**, transforming clinical OCT data and physician annotations into structured, model-ready volumetric inputs.
- **Designed models through controlled architecture progression** — from a shared 3D CNN backbone to graph message passing and Graph Structural Filtering — rather than comparing unrelated architectures.
- **Connected CNN and graph representations in one segmentation system**, using compact graph construction, long-range message passing, and graph-frequency feature reweighting.
- **Built a vessel-level validation framework** with repeated test sets, overlapping prediction aggregation, multi-metric evaluation, and paired model comparisons.
- **Used model behavior and failure analysis to guide conclusions**, distinguishing overall overlap, false-positive control, lesion coverage, and remaining image-quality limitations.

---

## Tech Stack

| Area | Technologies / Methods |
|---|---|
| **Programming & Modeling** | Python · TensorFlow · Keras · MATLAB |
| **Deep Learning** | 3D U-Net · Attention U-Net · Conv3D · GroupNorm · ASPP |
| **Graph Learning** | KNN Graph · GCN · GraphConv · Graph Laplacian · Graph Fourier Transform · Frequency SE |
| **Image Processing** | DICOM Processing · Polar Transformation · Artifact Removal · Signal Enhancement · Data Augmentation |
| **Experimentation** | Overlapping Prediction Aggregation · Vessel-level Evaluation · 3D Dice · Precision · Recall · IoU · Paired Statistical Testing |

---

## Context / Disclaimer

**Project Context**  
This repository summarizes my master's research in biomedical engineering at National Taiwan University, focused on 3D deep learning and graph-based modeling for coronary OCT calcification segmentation.

**Data Privacy**  
Raw clinical OCT data, patient-identifiable information, and physician annotation files are not included in this repository.

**Repository Scope**  
This portfolio contains sanitized technical descriptions, architecture designs, aggregated experimental results, and publication-safe examples. Restricted research code and non-public clinical materials are intentionally omitted.
