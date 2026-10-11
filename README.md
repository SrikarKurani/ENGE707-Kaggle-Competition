# ENGE707 Assessment 3 – Facial Feature Classification

This project involves classifying images of cat and dog facial features into eight classes: `CatEye`, `DogEye`, `CatFace`, `DogFace`, `CatMouth`, `DogMouth`, `CatNose`, and `DogNose`.

## Repository Structure

```text
data/             # Kaggle dataset (not committed to Git)
  train/          # Labelled training images
  test/           # Unlabelled images for Kaggle submission
src/
  config.py       # Dataset paths, class names, random seed and settings
  explore_data.py # Task 1: Dataset exploration
outputs/          # Figures, logs, models and submission files
```

## Setup

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition dataset from Kaggle and extract it into the `data/` folder.

## Running

Run the following command to explore the dataset:

```bash
python src/explore_data.py
```

This script checks the number of images in each class, displays sample images, and checks image sizes and quality.


## Kaggle

Username: srikarkurani