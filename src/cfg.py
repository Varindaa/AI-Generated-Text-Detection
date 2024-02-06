import keras


class CFG:
    verbose = 0  # Verbosity

    wandb = True  # Weights & Biases logging
    _wandb_kernel = 'awsaf49'  # WandB kernel
    comment = 'DebertaV3-MaxSeq_200-ext_s-torch'  # Comment description

    preset = "deberta_v3_base_en"  # Name of pretrained models
    sequence_length = 200  # Input sequence length

    device = 'TPU'  # Device

    seed = 42  # Random seed

    num_folds = 5  # Total folds
    selected_folds = [0, 1]  # Folds to train on

    epochs = 3  # Training epochs
    batch_size = 3  # Batch size
    drop_remainder = True  # Drop incomplete batches
    cache = True  # Caches data after one iteration, use only with `TPU` to avoid OOM

    scheduler = 'cosine'  # Learning rate scheduler

    class_names = ["real", "fake"]  # Class names [A, B, C, D, E]
    num_classes = len(class_names)  # Number of classes
    class_labels = list(range(num_classes))  # Class labels [0, 1, 2, 3, 4]
    label2name = dict(zip(class_labels, class_names))  # Label to class name mapping
    name2label = {v: k for k, v in label2name.items()}  # Class name to label mapping


if __name__ == '__main__':
    keras.utils.set_random_seed(CFG.seed)
