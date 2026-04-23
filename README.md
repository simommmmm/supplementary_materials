# Supplementary Materials

This supplementary package accompanies the paper and provides the main resources underlying the study, with particular focus on the final annotation codebook and the derived datasets.

## Contents

### 1. `codebook_final.pdf`
Final annotation codebook used in the study. It includes:
- the full set of annotation labels
- their operational definitions
- inclusion and exclusion criteria
- temporal thresholds where applicable
- the reduced set of final labels used for baseline modeling

### 2. `dataset_overview.pdf`
Overview of the released datasets and their construction pipeline. It summarizes:
- the distinction between CAM1, CAM2, and multi-view 3D datasets
- the temporal windowing strategy
- the final target labels
- the selection criteria used to build the ML-ready subsets

### 3. `data_dictionary.csv`
Column-level description of the released datasets, including:
- variable name
- type
- description
- dataset applicability

### 4. `thematic_analysis.pdf`
Document reporting the thematic analysis table derived from the educator interviews and used to support the co-design process.

### 5. `datasets/`
Derived tabular datasets used for baseline modeling.

#### `datasets/cam1/`
Person-level 2D feature dataset derived from CAM1.

#### `datasets/cam2/`
Person-level 2D feature dataset derived from CAM2.

#### `datasets/multiview_3d/`
Final person-level multi-view 3D datasets derived from real triangulation and controlled fusion.

### 6. `results/`
Summary files reporting the main baseline results for the released datasets.

## Notes on data format

- Each row in the released datasets corresponds to a person-level temporal window.
- The datasets do not contain raw videos or directly identifying information.
- Annotated identities are pseudonymous and used only as internal linking variables across derived representations.
- The released files are derived feature representations designed for analysis and baseline modeling.

## Ethical note

The supplementary materials do not include raw audiovisual recordings. The released datasets are limited to derived, non-biometric representations consistent with the privacy-preserving design of the project.

## Relationship between files

- The annotation codebook defines the labels.
- The dataset overview explains how labels and features were converted into trainable datasets.
- The data dictionary explains the meaning of each column in the released CSV files.