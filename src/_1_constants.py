if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import torch

TIFF_MAX_INTENSITY_VALUE = 65535.0
PYTORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EVALUATION_SPECIMEN_NAMES = {
    "fold 1": "14MoSiO2",
    "fold 2": "BiNi12-DSC",
    "fold 3": "CeO2Ag",
    "fold 4": "nanowires-PLA, brass",
    "fold 5": "nanowires-PLA, Coal",
    "fold 6": "nanowires-PLA, full-headlights",
    "fold 7": "nanowires-PLA, SBA-15"
}


# general training hyperparameters
PATCH_SIZE_PX = 256
PATCHES_PER_BATCH = 128
BATCHES_PER_EPOCH = 32
TOTAL_EPOCHS = 4000
START_LEARNING_RATE = 1e-4


# pre-training hyperparameters - loss component weights per stage
LOSS_COMPONENT_WEIGHTS_PRETRAINING = {
    "rec":                {1: 1.0,      2: 10.0,      3: 50.0        },
    "tv_geo":             {1: 2.0,      2: 10.0,      3: 0.0         },
    "tv_c":               {1: 2.0,      2: 10.0,      3: 100.0       },
    "sphere":             {1: 1.0,      2: 1.5,       3: 1.5         },
    "comp_uni":           {1: 1000.0,   2: 100.0,     3: 100.0       },
    "eta_inter_det_corr": {1: 5000.0,   2: 100_000.0, 3: 10_000_000.0},
    "white":              {1: 10_000.0, 2: 100_000.0, 3: 10_000_000.0},
    "t1_p_mag":           {1: 0.5,      2: 1.0,       3: 0.0         },
    "t1_alpha":           {1: 0.5,      2: 1.0,       3: 0.0         },
    "t2_alpha":           {1: 0.5,      2: 1.0,       3: 0.0         },
    "r_low_reg":          {1: 1.0,      2: 1.0,       3: 2.0         }
}


# finetuning hyperparameters - loss component weights
LOSS_COMPONENT_WEIGHTS_FINETUNING = {
    "rec":       50.0,
    "c_match":   100.0,
    "n_match":   100.0,
    "r_low_reg": 2.0,
    "white":     10_000.0
}
