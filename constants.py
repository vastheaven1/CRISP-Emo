"""CRISP-Emo signal constants."""

EEG_CHANNELS = (
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
)

SHARED14_EEG_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)

PERIPHERY10_EEG_CHANNELS = (
    "AF3", "AF4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2",
)

EEG_MONTAGES = {
    "full32": tuple(range(len(EEG_CHANNELS))),
    "shared14": tuple(EEG_CHANNELS.index(name) for name in SHARED14_EEG_CHANNELS),
    "periphery10": tuple(EEG_CHANNELS.index(name) for name in PERIPHERY10_EEG_CHANNELS),
}

TARGET_RATES = {"eeg": 128, "gsr": 32, "skt": 8}
BASELINE_SECONDS = 5.0
DEFAULT_WINDOW_SECONDS = 5.0
DREAMER_EEG_RATE = 128
DREAMER_ECG_RATE = 256
