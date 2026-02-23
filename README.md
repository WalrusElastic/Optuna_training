# Optuna YOLO Training Pipeline

## Overview

This is an **Optuna-based hyperparameter optimization pipeline** for training YOLO11 segmentation models. The pipeline automatically runs multiple training trials with different hyperparameters, evaluates them on validation and test datasets, and identifies the best-performing configuration.

The pipeline is designed to:
- **Optimize custom hyperparameters** (e.g., brightness, contrast, sharpness, mosaic ratio, mixup) using Optuna's Bayesian optimization
- **Automatically track and compare trials** across multiple runs
- **Generate comprehensive metrics** including per-class precision, recall, and mAP scores
- **Produce visualizations** such as loss curves and confusion matrices
- **Save detailed results** in CSV and JSON formats for analysis

## Quick Start

### Prerequisites

1. **Python 3.10+** with PyTorch and CUDA support
2. **Required packages**:
   ```bash
   pip install optuna ultralytics albumentations pyyaml pandas seaborn scikit-image shapely
   ```

### Setup

1. **Prepare your folder structure**:
   ```
   optuna_train_pipeline/
   ├── train.py                    (main script - modify optuna "objective" function for customised optimisation)
   ├── configs.py                  (configure paths and default parameters here)
   ├── data.yaml                   (generated automatically)
   ├── yolo11n-seg.pt             (your pretrained YOLO model weights)
   ├── split_dataset/             (input: train/val/test splits)
   │   ├── train/
   │   │   ├── images/
   │   │   └── labels/
   │   └── val/
   │       ├── images/
   │       └── labels/
   │   └── test/
   │       ├── images/
   │       └── labels/
   ├── Final_dataset/             (generated during preprocessing)
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── runs/                       (generated: trial outputs)
   ├── utils/                      (utility modules)
   │   ├── preprocessing_utils.py
   │   ├── yolo_dataset_utils.py
   │   ├── evaluation_utils.py
   │   ├── extract_yolo_data_utils.py
   │   └── optuna_utils.py
   └── README.md
   ```

2. **Configure `configs.py`**:
   ```python
   class TrainingConfig:
       def __init__(self):
           self.study_name = "my_study"          # Study name for tracking
           self.classes = ["dog", "cat", "bird"]  # Your class names
           
           # Paths (relative to configs.py location. Optional to configure)
           self.paths = {
               "default_model_weights": root / "yolo11n-seg.pt",
               "split_dataset": root / "split_dataset",
               "final_dataset": root / "Final_dataset",
               "test_dataset": (root / "Final_dataset") / "test",
               ...
           }
           
           # YOLO training default parameters
           self.yolo_parameters = {
               "epochs": 10,
               "batch": 8,
               "imgsz": 1024,
               "patience": 50,
               ...
           }
           
           # Additional parameters eg. for preprocessing
           self.additional_parameters = {
               "brightness": 0.3,
               "contrast": 0.3,
               "sharpness": 0.5,
           }
   ```

3. **Customize which parameters to optimize in `train.py`**:

   In the `objective()` function, uncomment the parameters you want Optuna to optimize:
   ```python
   def objective(trial: optuna.trial.Trial, config: TrainingConfig) -> float:
       # ...
       
       # Uncomment parameters to optimize:
       brightness = trial.suggest_float("brightness", 0.2, 0.5)
       contrast = trial.suggest_float("contrast", 0.1, 0.4)
       sharpness = trial.suggest_float("sharpness", 0.3, 0.8)
       
       # Update relevant storage dictionary with suggested values
       additional_parameters["brightness"] = brightness
       additional_parameters["contrast"] = contrast
       additional_parameters["sharpness"] = sharpness
       
       # You can also optimize YOLO parameters:
       # mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
       # params["mosaic"] = mosaic
       
       # ... rest of function
   ```

### Running the Pipeline

```bash
python train.py
```

The script will:
1. **Preprocess** the dataset (augmentation, splits)
2. **Train** the YOLO model for one trial
3. **Evaluate** on validation and test sets
4. **Extract metrics** per class and overall
5. **Save results** to CSV, JSON, and visualizations

### Single Trial vs. Multiple Trials

Currently, the pipeline runs **1 trial at a time** for testing purposes. To run multiple optimization trials:

```python
# In train.py, modify the optimize call:
study.optimize(
    lambda trial: objective(trial, config),
    n_trials=10,  # Run 10 trials
    n_jobs=1
)
```

## Output Files

After each trial, you'll find:

- **`runs/trial_0/`** - Trial outputs directory
  - `weights/best.pt` - Best model weights
  - `results.csv` - Training metrics per epoch
  - `test_results/` - Predictions and confusion matrix

As the trials progress, you will also find:

- **`study_name_output.csv`** - All trials comparison (append mode)
  - Columns: hyperparameters, validation metrics, test metrics
  
- **`study_name_output.json`** - Detailed per-trial results
  
- **`study_name_optuna_storage.json`** - Optuna study state (for resuming)

## Optimization Metrics

The pipeline optimizes based on:
- **Custom scoring**: Currently uses `class_2_mAP50-95` (modify in `objective()`)

## Customization

### Add New Optimization Parameters

1. In `objective()` function, add a `trial.suggest_*()` call:
   ```python
   my_param = trial.suggest_float("my_param", min_val, max_val)
   ```

2. Use the suggested value in your pipeline:
   ```python
   params["my_param"] = my_param
   ```

3. Results will automatically be logged in CSV and JSON outputs

### Change Evaluation Metric

Modify the scoring function in `objective()`:
```python
# Current: class 2 mAP50-95
score = round(float(results.box.class_result(2)[3]), 5)

# Alternative: average mAP across all classes
class_maps = [results.box.class_result(i)[3] for i in range(len(config.classes))]
score = round(float(np.mean(class_maps)), 5)
```


## Workflow Example

1. **Setup once**:
   - Place model weights: `yolo11n-seg.pt`
   - Place dataset: `split_dataset/train/images/`, `split_dataset/val/images/`, etc.
   - Edit `configs.py` with your class names and paths
   - Choose parameters to optimize in `train.py`

2. **Run optimization**:
   ```bash
   python train.py
   ```

3. **Run multiple times** (to accumulate trials):
   - Repeat step 2; results append to CSV/JSON
   - Optuna tracks best trial automatically

4. **Analyze results**:
   - Open `study_name_output.csv` in Excel/pandas
   - Compare hyperparameters vs. metrics
   - Identify best configuration

## Troubleshooting

**CUDA Out of Memory**
- Reduce `batch` size in `configs.py`
- Reduce `imgsz` (image size)

**Trials failing with tensor errors**
- Check dataset format (must be YOLO segmentation format)
- Ensure labels match image dimensions

**Optuna not trying new parameters**
- Check that `trial.suggest_*()` calls are uncommented in `objective()`
- Verify parameter ranges are reasonable

**Results not updating**
- Check that output paths exist: `runs/`, `Final_dataset/`
- Verify write permissions on CSV/JSON files

## File Structure Details

| File | Purpose |
|------|---------|
| `train.py` | Main training script (run this) |
| `configs.py` | All configuration: paths, classes, parameters |
| `utils/preprocessing_utils.py` | Dataset augmentation and preprocessing |
| `utils/yolo_dataset_utils.py` | Custom YOLO dataset loader with class weighting |
| `utils/evaluation_utils.py` | Segmentation metrics and confusion matrix |
| `utils/extract_yolo_data_utils.py` | Extract and format training outputs |
| `utils/optuna_utils.py` | Optuna study management |
 

For issues or questions, check the script comments in `objective()` function for parameter-specific guidance.
