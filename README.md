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
| **4. Model Design** | Standard 3D CNNs mainly capture local patterns and may miss wider structural relationships | Developed and compared **3D U-Net, attention, GCN, graph structural filtering, gated fusion, and non-local** architectures | Identified **GSF_GAP** as the strongest overall model with mean **3D Dice 0.834** | Deep Learning, 3D CNN, GNN, Architecture Design |
| **5. Evaluation & Results** | Model improvements needed to be validated across full vessels and repeated experiments rather than individual VoIs or one favorable split | Reconstructed overlapping VoI predictions into full-vessel masks and evaluated models using **Dice, Precision, Recall, IoU, and statistical testing** | Achieved best mean **Dice 0.834** and **IoU 0.703**, while identifying precision–recall trade-offs across architectures | Experiment Design, Statistical Testing, Model Evaluation |

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

**Architecture / Workflow**

```mermaid
flowchart LR
    A["OCT Challenges"] --> B["Weak Signals & Artifacts"]
    A --> C["Limited 2D Context"]
    A --> D["Complex Boundaries"]

    B --> E["Robust Preprocessing"]
    C --> F["3D Input Design"]
    C --> G["Graph-based Modeling"]
    D --> G
    D --> H["Multi-metric Evaluation"]

    E --> I["End-to-End Segmentation Pipeline"]
    F --> I
    G --> I
    H --> I
```

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

> **Design Goal:** Extend a strong 3D segmentation backbone from local image modeling to explicit long-range graph reasoning, then improve graph message passing through adaptive graph-frequency filtering.

### Architecture Evolution

The model family was designed as a sequence of technical questions rather than a collection of independent architectures.

| Step | Limitation to Address | Design Response | Model |
|---|---|---|---|
| **1. Build strong 3D features** | Calcification extends across slices and varies in scale | 3D convolution, multi-scale context, depth-aware receptive field, attention-guided decoding | **U-Net** |
| **2. Add long-range relations** | CNN features interact mainly through local convolution | Convert multi-scale features into graph nodes and perform KNN-based message passing | **GCN U-Net** |
| **3. Improve graph information** | Feature-similar KNN connections may contain irrelevant or noisy structural relationships | Decompose graph signals into frequency components and adaptively reweight them before message passing | **GSF_GAP U-Net** |
| **4. Test GSF design choices** | Frequency importance can be described or filtered in different ways | Compare GAP, RMSP, and learnable soft gating | **GSF_RMSP / Gated GSF_GAP** |
| **5. Benchmark other strategies** | Improvements should be separated from generic attention or global-context effects | Compare with image-space feature weighting and graph-free dense global attention | **CWA / Non-local U-Net** |

The final design therefore follows:

> **3D Image Modeling → Graph Relation Modeling → Graph Structural Filtering**

---

### Model Family

```mermaid
flowchart LR
    A["3D Attention U-Net<br/>Image Baseline"]

    A -->|"Need explicit<br/>long-range relations"| B["GCN U-Net<br/>Graph Baseline"]

    B -->|"Graph information<br/>may contain noise"| C["GSF_GAP U-Net<br/>Main Method"]

    C --> D["GSF_RMSP<br/>Frequency Descriptor Variant"]
    C --> E["Gated GSF_GAP<br/>Soft-filtering Variant"]

    A -. "Alternative feature modeling" .-> F["CWA U-Net"]
    A -. "Alternative global modeling" .-> G["Non-local U-Net"]
```

Each model has a clear role:

- **U-Net** — 3D image baseline
- **GCN U-Net** — graph message-passing baseline
- **GSF_GAP U-Net** — main graph-frequency filtering method
- **GSF variants** — test alternative frequency weighting and gating designs
- **CWA / Non-local U-Net** — benchmark alternative feature and global-context modeling strategies

---

### 1. Backbone Configuration Selection

Before adding graph modules, I first fixed a common backbone configuration to keep downstream comparisons controlled.

| Component | Previous | Selected |
|---|---:|---:|
| **Input depth** | 5 slices | **8 slices** |
| **Base channels** | 8 | **32** |
| **Normalization** | BatchNorm | **GroupNorm** |

The selected configuration was then **held fixed across all downstream models**, isolating the impact of graph and spectral modules.

---

### 2. Shared 3D Backbone

All downstream models use the same **3D Attention U-Net feature extractor**.

- Input: `256 × 256 × 8 × 1`
- Base channels: `32`
- Group Normalization
- ASPP with dilation rates `3 / 6 / 9`
- Depth-dilated 3D convolution
- Attention-gated skip connections

> **Role:** provide a fixed 3D feature backbone for controlled architecture comparisons.

---

### 3. Graph Representation & Message Passing

To model long-range dependencies beyond local convolution, multi-scale CNN features are projected into a compact graph representation.

```text
Multi-scale Features
        ↓
Graph Pooling
        ↓
4×4×8 = 128 Nodes
        ↓
KNN Graph (k = 4)
        ↓
2× GraphConv
        ↓
Shared Decoder
```

Graph pooling reduces computation while preserving the full 8-slice depth structure.

> **Role:** explicitly model long-range feature interactions through graph message passing.

---

### 4. Graph Structural Filtering

KNN graphs can connect feature-similar regions that are not equally useful for segmentation.

GSF adds **spectral feature reweighting before GraphConv**:

```text
Node Features
     ↓
Graph Fourier Transform
     ↓
Frequency Reweighting
     ↓
Residual Fusion
     ↓
GraphConv
```

The main model, **GSF_GAP U-Net**, uses Frequency Squeeze-and-Excitation to learn the contribution of graph-frequency components before message passing.

> **Role:** refine graph features before propagation instead of treating all graph signals equally.

---

### 5. GSF Ablation Variants

| Model | Design Change |
|---|---|
| **GSF_GAP** | Global Average Pooling for frequency descriptors |
| **GSF_RMSP** | RMS pooling to capture frequency magnitude |
| **Gated GSF_GAP** | Learnable soft gate for additional frequency suppression |

These ablations isolate the effect of **frequency descriptor design** and **soft gating** while keeping the rest of the graph pipeline fixed.

---

### 6. Comparison Baselines

Two additional baselines benchmark alternative context-modeling strategies:

- **CWA U-Net** — image-space feature reweighting
- **Non-local U-Net** — dense global attention without graph construction

Together with U-Net and GCN U-Net, they provide controlled reference points for evaluating the proposed GSF models.

---

### Design Summary

> **3D Feature Extraction → Graph Message Passing → Spectral Reweighting**

The architecture progressively adds **3D context, explicit long-range dependencies, and frequency-aware graph feature selection** while keeping the shared segmentation backbone fixed.

[Model Design Deep Dive →](docs/04_model_design.md)

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
