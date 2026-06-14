import logging
import albumentations as A
import typing
import shutil
from pathlib import Path    
import yaml

from utils.preprocessing_utils import PreprocessingUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrepareTraining:
    '''
    Sets up YOLO data yaml, prepares and augments final dataset and applies Optuna suggestions to training parameters.
    '''
    @staticmethod
    def prepare_augmented_dataset(
    config: typing.Any,
    trial_number: int,
) -> None:
        """
        Applies preprocessing, augmentations and upscalingto the Split Dataset, writing it in YOLO format.
        
        Example split dataset format:
            Split_dataset/
            ├── train/
            │   ├── image1.tif
            │   └── image1.txt
            ├── valid/
            │   ├── image1.tif
            │   └── image1.txt
            └── test/ (optional)
                ├── image1.tif  
                └── image1.txt
                
        Args:
            config (TrainingConfig): Config object from config loader
            trial_number (int): Current trial number (for logging)
        Output:
            Writes augmented dataset to Final_dataset/ or filepath specified in config
        """
        logger.info(f"[Trial {trial_number}] Starting dataset preparation")
        
        # Define augmentation transforms
        brightness = config.augmentation_parameters["brightness"]
        contrast = config.augmentation_parameters["contrast"]
        sharpness = config.augmentation_parameters["sharpness"]

        transforms = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(contrast, contrast),
                brightness_by_max=False,
                p=1
            ),
            A.Sharpen(alpha=(sharpness, sharpness), lightness=(0.8, 0.8), p=1.0),
        ])

        # Augment and prepare dataset
        split_path = config.paths["split_dataset"]
        final_path = config.paths["final_dataset"]
        
        logger.info(f"[Trial {trial_number}] Augmenting dataset")
        if final_path.exists():
            shutil.rmtree(final_path)

        input_size = config.preprocessing_config["input_size"]
        output_size = config.preprocessing_config["training_size"]
        edit_labels = config.preprocessing_config["edit_labels"]

        for split in ['train', 'valid', 'test']:
            src_dir = split_path / split
            if not src_dir.exists():
                logger.warning(f"[Trial {trial_number}] Skipping {split} split - not found")
                continue

            logger.info(f"[Trial {trial_number}] Processing {split} split")
            images_dir = final_path / split / "images"
            labels_dir = final_path / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            tif_files = list(src_dir.glob("*.tif"))
        
            logger.info(f"[Trial {trial_number}] Found {len(tif_files)} images in {split}")
            
            for tif_path in tif_files:
                txt_path = tif_path.with_suffix('.txt')
                
                PreprocessingUtils.generate_transform(
                    str(tif_path),
                    str(txt_path),
                    str(images_dir),
                    str(labels_dir),
                    transforms,
                    edit_labels=edit_labels,
                    iterations=1,
                    input_img_size=input_size,
                    output_img_size=output_size,
                )

        logger.info(f"[Trial {trial_number}] Dataset preparation completed")

    @staticmethod
    def setup_data_yaml(yaml_path: Path, dataset_path: Path, classes: list) -> None:
        """Create data.yaml for RF-DETR dataset (in YOLO format).
        Args:
            yaml_path: Path to save data.yaml (in Final_dataset folder)
            dataset_path: Path to dataset (root containing train/valid/test)
            classes: List of class names
        """
        logger.info(f"Setting up YOLO data.yaml at {yaml_path}")
        data_yaml = {
            "path": str(dataset_path),
            "train": "train",
            "val": "valid",
            "test": "test",
            "augment": True,
            "channels": 1,
            "names": {index: name for index, name in enumerate(classes)}
        }

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_yaml, f)

        logger.info(f"data.yaml created with {len(classes)} classes")

    @staticmethod
    def applyOptunaSuggestions(trial, config):
        training_params = config.rfdetr_parameters.copy()
        
        all_params = training_params.copy()
        all_params.update(config.preprocessing_config)
        # Apply Optuna search-space suggestions (overrides training_params)
        search_space = getattr(config, "optuna_search_space", {})
        logger.info(f"[Trial {trial.number}] Search space: {search_space}")
        for name, spec in search_space.items():
        
            param_type = spec.get("type", "float")
            if param_type == "float":
                val = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False))
                )
            elif param_type == "int":
                if "step" in spec:
                    val = trial.suggest_int(
                        name,
                        int(spec["low"]),
                        int(spec["high"]),
                        step=int(spec["step"])
                    )
                else:
                    val = trial.suggest_int(name, spec["low"], spec["high"])
            elif param_type == "categorical":
                val = trial.suggest_categorical(name, spec["choices"])
            else:
                continue
            
            training_params[name] = val
            logger.info(f"[Trial {trial.number}] Optuna suggestion: {name} = {val}")

        return training_params, all_params
    
    @staticmethod
    def prepare_trial(trial, config):
        trial_number = trial.number
        logger.info(f"[Trial {trial_number}] Preparing trial")
        
        # Step 1: Prepare augmented dataset
        PrepareTraining.prepare_augmented_dataset(config, trial_number)

        # Step 2: Setup data.yaml
        PrepareTraining.setup_data_yaml(
            yaml_path=config.paths["data_yaml"],
            dataset_path=config.paths["final_dataset"],
            classes=config.classes
        )

        logger.info(f"[Trial {trial.number}] Preparing training parameters")
        # Step 3: Apply Optuna suggestions to training parameters
        training_params, all_params = PrepareTraining.applyOptunaSuggestions(trial, config)

        return training_params, all_params