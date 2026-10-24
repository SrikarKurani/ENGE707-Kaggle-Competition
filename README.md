# ENGE707 Assessment 3 – Facial Feature Classification

This project classifies images of cat and dog facial features into eight classes: `CatEye`, `DogEye`, `CatFace`, `DogFace`, `CatMouth`, `DogMouth`, `CatNose` and `DogNose`.

The final model is a pretrained EfficientNet-B0 that was fine-tuned on the competition dataset. It achieved a score of **0.9783** on the Kaggle public leaderboard.

## Results

- **EfficientNet-B0 (final model):** 99.40% validation accuracy, 0.9783 Kaggle score
- **ResNet-18:** 99.24% validation accuracy, 0.9743 Kaggle score

The most common error for both models is `CatNose` being predicted as `CatMouth` (41 validation images for EfficientNet-B0 and 45 for ResNet-18).

## Figures

- **Dataset exploration:** [Class distribution](outputs/figures/class_distribution.png), [sample images](outputs/figures/sample_grid.png), [image sizes](outputs/figures/size_distribution.png)
- **EfficientNet-B0:** [Learning curves](outputs/figures/learning_curves_efficientnet_b0.png), [confusion matrix](outputs/figures/confusion_matrix_efficientnet_b0.png)
- **ResNet-18:** [Learning curves](outputs/figures/learning_curves_resnet18.png), [confusion matrix](outputs/figures/confusion_matrix_resnet18.png)
- **Kaggle evidence:** [Leaderboard](outputs/kaggle_leaderboard.png), [submissions](outputs/kaggle_submissions_both.png)

## Repository Structure

```text
data/                 # Kaggle dataset (not committed to Git)
  train/train/        # Labelled training images (00001.jpg, ...)
  test/test/          # Unlabelled test images for Kaggle predictions
  train_labels.csv    # Image IDs and their class labels
  sample_submission.csv
src/
  config.py           # Dataset paths, class names, random seed and settings
  explore_data.py     # Task 1: dataset exploration and figures
  dataset.py          # Task 2: preprocessing, data split and data loaders
  train.py            # Tasks 3-4: model training, early stopping and learning curves
  evaluate.py         # Tasks 5-6: validation metrics, confusion matrix and common errors
  predict.py          # Task 5: generate the Kaggle submission file
outputs/
  figures/            # Figures used in the report
  history_*.csv       # Training and validation results for each epoch
  classification_report_*.csv
  submission.csv      # Final EfficientNet-B0 Kaggle submission
  submission_resnet18.csv
  models/             # Saved model weights (not committed)
```

## Setup

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition dataset from Kaggle and extract it into the `data/` folder. Make sure the files follow the directory structure shown above.

## Running the Pipeline

Run the following commands from the main project folder, in order:

```bash
python src/explore_data.py                     # Explore the dataset and check image quality
python src/dataset.py                          # Check the data split and data loaders
python src/train.py --model efficientnet_b0    # Train the model and save the best version
python src/evaluate.py --model efficientnet_b0 # Evaluate the model on the validation set
python src/predict.py --model efficientnet_b0  # Generate outputs/submission.csv
```

To train the comparison model, replace `efficientnet_b0` with `resnet18` in the relevant commands.

A fixed random seed of 42 is used to keep the training and validation split consistent between runs.

### Training on Kaggle

Training was carried out in a Kaggle notebook using a Tesla T4 GPU, as training on a CPU takes much longer. The notebook clones the repository, links the competition dataset to the `data/` folder and runs the same scripts used locally.

```python
!git clone https://github.com/SrikarKurani/ENGE707-Kaggle-Competition.git
%cd ENGE707-Kaggle-Competition
!ln -s "$(dirname "$(find /kaggle/input -name train_labels.csv | head -1)")" data
!python -u src/train.py --epochs 10
!python src/evaluate.py
!python src/predict.py
```

After the images were cached, each epoch took approximately 3 minutes for EfficientNet-B0 and 2 minutes for ResNet-18.

## Kaggle

**Username:** srikarkurani (display name: Srikar Kurani)
