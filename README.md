
# RFDETR Optuna Training Manual



## Overview

This is an **Optuna-based hyperparameter optimization pipeline** for training YOLO11 segmentation models. The pipeline automatically runs multiple training trials with different hyperparameters, evaluates them on validation and test datasets, and identifies the best-performing configuration. <br>

The pipeline is designed to: <br>

1. **Optimize custom hyperparameters** (e.g., brightness, contrast, sharpness, mosaic ratio, mixup) using Optuna's Bayesian optimization

2. **Generate comprehensive metrics** including per-class precision, recall, and mAP scores
3. **Automatically track and compare trials** across multiple runs
4. **Save detailed results** in CSV and JSON formats for analysis
<br>
## Quick Start

### Prequisites

1. **Python 3.10+** with PyTorch ad CUDA support
2. **Required packages** (and hope you venv is blessed)
``` 
pip install -r requirements.txt
```

### Setup
1. **Prepare your folder structure**
>```
> optuna_train_pipeline/
>├── rf_detr_train.py             (main script - modify optuna "objective" function for customised optimisation)
>├── config_loader.py                  
>├── config.yaml                  (configure paths and default >parameters here)
>├── preparetraining.py
>├── rf_detr_threshold_sweep.py
>├── cli.py
>├── rf-detr-nano.pth             (your pretrained rf-detr weights)
>├── split_dataset/             (input: train/val/test splits (in yolo seg format))
>│   ├── train/
>│   │   ├── image1.tif              (note image and labels together in 1 folder)
>│   │   └── label1.txt
>│   └── val/
>│   │   ├── image1.tif
>│   │   └── label1.txt
>│   └── test/
>│   │   ├── image1.tif
>│   │   └── label1.txt
>├── utils/                      (utility modules)
>│   ├── preprocessing_utils.py
>│   ├── data_logging_utils.py
>│   ├── dataset_converter.py
>│   ├── rf_detr_distributed_worker.py
>│   ├── rf_detr_extract_utils.py
>│   ├── rf_detr_prediction_utils.py
>│   ├── rf_detr_trainer.py
>│   ├── yolo_evaluation_utils.py
>│   ├── yolo_dataset_utils.py
>│   └── optuna_utils.py
>└── README.md
>```

2. **Configure** ```config.yaml```:

A template of the yaml is included in the repo. Note that **filepaths should be relative to** ```rf-detr-train.py``` and be written as strings. After editing the study **name** and **classes**, the key sections to configure would be: 
- paths
- preprocessing
- rfdetr_training
- optuna

```yaml
# Optuna RF-DETR Training Configuration
study:
  name: "test_study"

classes:
  - "Class_1"

paths:
    pretrained_model_weights: "rf-detr-nano.pth"
```

    
&emsp;&emsp;2.1 **Paths**

Below are the paths that you will probably edit between experiments. You can set other paths if needed, but they should be left to default values. Ensure that **split_dataset** is in the right format and location. It should contain the images and labels for training, **before any preprocessing**.
```yaml
paths:
  pretrained_model_weights: "rf-detr-nano.pth"
  split_dataset: "split_dataset"              # original INPUT dataset in yolo format (see section 1)
  final_dataset: "Final_dataset"              # augmented dataset, wil be generated automatically
  runs_dir: "runs"                            # folder where runs are stored
  params_json: "training_params.json"         # json where training parameters for the run is stored
```


&emsp;&emsp;2.2 **Preprocessing**

As of now these are the necessary inputs, used to determine upscale factor. When fine tuning of preprocessing strength is implemented, the parameters will be input here.

Note that **augmentations** (via albumentations) has its own section, but are not necessary to configure (default = 0).
```yaml
preprocessing:
  input_size: 256         # Size of raw image slices (in split dataset)
  training_size: 1024      # Size of upscaled images, which the model will train on (4x upscaling)
```

&emsp;&emsp;2.3 **rfdetr_training**

These are the commonly edited training parameters. ```dataset_dir``` is the folder containing the final **augmented** dataset, which rfdetr will train on, and defaults to ```Final Dataset```. Also note that ```aug_config``` refers to the **default rfdetr augmentations** which are imported. Simply enter the string version of whichever set of augmentations you wish to apply.

The rest of the common hyperparameters (lr, dropout, weight decay) should be implemented here. Check the template for the full list of configurable parameters. 
```yaml
rfdetr_training:
  dataset_dir: None  # Defaults to Final Dataset path if None
  batch_size: 1
  grad_accum_steps: 4
  epochs: 1
  early_stopping_patience: 1
  aug_config: "AUG_AGGRESSIVE" #String version of rf-detr augmentation config.
```
&emsp;&emsp;2.4 **Optuna**

This is where you can configure the settings for Optuna. This mainly includes:

- n_trials: Number of trials that optuna will run
- search_space: Parameters which optuna will optimise

In the example below, ```lr``` is being tuned. The configs follow the optuna ```.suggest``` arguments, usually the min and max values that you want optuna to try. This includes step size for ```int```. 

```yaml
optuna:
  n_trials: 1
  n_jobs: 1
  optimize: false  # Set to true to use Optuna hyperparameter optimization
  search_space:  #parameter for optuna to optimise. currently limited to rf-detr params.
    lr:
      type: float
      low: 1e-5
      high: 1e-3
      log: true
```
## Running the Pipeline

    python rf_detr_train.py

The script will run an optuna pipeline to optimise model performance accross a set of trials. Each trial will:

1. **Suggest** Parameters for optimization
2. **Preprocess** the dataset (augmentation, splits)
3. **Train** the rfdetr model for one trial
4. **Evaluate** on validation and test sets
5. **Extract metrics** per class and overall
6. **Save results** to CSV, JSON, and visualizations

You can also input arguments via the CLI, if you want to quickly run trials with other parameters. Eg to run 200 epochs:

    python rf_detr_train.py --epochs 200

Use ```--help``` to see a full list of CLI arguments. 

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
- You can change the class to optimise in the configs

## Customization

### Add New Optimization Parameters

**THIS IS BEING UPDATED TO BE FULLY MODULAR VIA THE YAML -@Z113x**

1. In `config.yaml`, add a `parameter` under search space:
   ```yaml
     search_space:
        new_params:
          type: int
          low: 1
          high: 10
          step: 2
        new_param_2:
          type: float
          ... rest of params
   ```

2. The parameter should be automatically updated if it is part of ``rfdetr parameters`` in the yaml. As of 14/6/2026, preprocessing and augmentations tuning are not automatically supported. 
<br> 
To debug, check that ``applyOptunaSuggestions()`` in ``preparetraining.py`` supports your data type. 
   ```python
      if param_type == "float":  # For tuning floats
      val = trial.suggest_float(
          name,
          float(spec["low"]),
          float(spec["high"]),
          log=bool(spec.get("log", False))
      )
      elif param_type == "int": # For tuning ints
          if "step" in spec:
              val = trial.suggest_int(
                  name,
                  int(spec["low"]),
                  int(spec["high"]),
                  step=int(spec["step"])
              )
          else:
            #... rest of data types
   ```
3. log results in appropriate dictionary to be captured in CSV and JSON outputs

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
   - Place model weights: `rf-detr-nano.pth`
   - Place dataset: `split_dataset/train/`, `split_dataset/val/`, etc.
   - Edit `configs.yaml` with your class names and paths
   - Choose parameters to optimize in `train.py`

2. **Run optimization**:
   ```bash
   python rf_detr_train.py
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

## File Structure Details (Outdated 14/6/2026)

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

