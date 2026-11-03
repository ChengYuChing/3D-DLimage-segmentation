# 3D Deep Learning — Medical AI Image Segmentation

**Coronary OCT segmentation with 3D deep learning and graph-based feature modeling.**

This repository is the main technical portfolio for my coronary OCT segmentation project. It is structured to show the full engineering workflow from real-world clinical imaging data to model design, controlled experiments, and engineering insights.

## Project Story

```text
Problem
  ↓
Data
  ↓
Preprocessing & 3D Volume Construction
  ↓
Model Design
  ↓
Experiments & Results
  ↓
Engineering Takeaways
```

## 01. Problem
What problem is being solved, why coronary OCT segmentation is technically challenging, and what the project is designed to improve.

→ [Read more](docs/01_problem.md)

## 02. Data
Dataset scale, image characteristics, labeling setup, and how the raw OCT data are organized before model development.

→ [Read more](docs/02_data.md)

## 03. Pipeline
The preprocessing and input-construction workflow that converts raw OCT frames into model-ready 3D volumes.

→ [Read more](docs/03_pipeline.md)

## 04. Model Design
The evolution from a 3D U-Net baseline to attention-, graph-, spectral-, gated-, and non-local variants.

→ [Read more](docs/04_model_design.md)

## 05. Experiments & Results
Experimental protocol, evaluation metrics, repeated validation, statistical testing, quantitative results, qualitative comparison, and failure cases.

→ [Read more](docs/05_experiments_results.md)

## 06. Engineering Takeaways
Key technical decisions, trade-offs, lessons learned, and what this project demonstrates from an ML engineering perspective.

→ [Read more](docs/06_engineering_takeaways.md)

## Repository Structure

```text
3D-DLimage-segmentation/
├── README.md
├── docs/
│   ├── 01_problem.md
│   ├── 02_data.md
│   ├── 03_pipeline.md
│   ├── 04_model_design.md
│   ├── 05_experiments_results.md
│   └── 06_engineering_takeaways.md
├── assets/
│   ├── overview/
│   ├── pipeline/
│   ├── architecture/
│   └── results/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── configs/
├── notebooks/
├── requirements.txt
├── LICENSE
└── .gitignore
```

> The repository is being organized for public release. Clinical data and restricted research materials will not be included.
