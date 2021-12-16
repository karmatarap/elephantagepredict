import torch
import librosa
import cv2
import numpy as np
import torch.nn as nn
from audiomentations import AddGaussianNoise
import tempfile


class AudioParams:
    """ Parameters specific to audio manipulations """

    # Wav params
    channel = 2
    duration = 15739
    sr = 22050

    # Spec params
    n_mels = 512
    fmin = 0
    fmax = sr // 2
    n_fft = sr // 10
    hop_length = sr // (10 * 4)

    # Image params
    resize = 256, 256  # 512,512


class ElephantDataset:
    def __init__(
        self, wav_paths, labels, params, wav_augmentations=None, spec_augmentations=None
    ):
        self.wav_paths = wav_paths
        self.labels = labels

        self.params = params

        self.wav_augmentations = wav_augmentations
        self.spec_augmentations = spec_augmentations

    def process(self, wav_file):

        # read wav
        y, sr = librosa.load(wav_file)

        # apply wav augmentations
        if self.wav_augmentations is not None:
            y = self.wav_augmentations(y, self.params.sr)

        # convert to mel spectrogram

        image = librosa.feature.melspectrogram(
            y,
            sr=self.params.sr,
            n_mels=self.params.n_mels,
            fmin=self.params.fmin,
            fmax=self.params.fmax,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
        )

        image = librosa.power_to_db(image).astype(np.float32)

        # Add Frequency masking, randomly mask rows and columns
        # of spectrogram and impute to mean value
        if self.spec_augmentations is not None:
            image = self.spec_augmentations(image)

        if self.params.resize is not None:
            image = ElephantDataset.resize(image, self.params.resize)

        # Spectrogram has no colour channel, required for CNNs
        image = ElephantDataset.spec_to_image(image)

        # Pytorch expects CxHxW format
        image = np.transpose(image, (1, 0, 2)).astype(np.float32)

        return image

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_file = self.wav_paths[idx]
        label = self.labels[idx]
        image = self.process(wav_file)

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )

    @staticmethod
    def spec_to_image(spec, eps=1e6):
        """ Convert mono to color by duplicating channels and normalizing """
        spec = np.stack([spec, spec, spec], axis=1)
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_norm = np.clip(spec_norm, spec_min, spec_max)
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    @staticmethod
    def resize(image, size=None):
        if size is not None:
            image = cv2.resize(image, size)
        return image


def load_model(seed=100):

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(f"app/models/best_model_params_{seed}.pt"))
    model.eval()
    return model


def predict(wav_path):
    wav_augs = AddGaussianNoise()
    dataset = ElephantDataset(
        [wav_path], [0, 0], AudioParams(), wav_augmentations=wav_augs
    )
    labels = ["ad/sa", "inf/juv"]
    with torch.no_grad():
        data = dataset[0]
        input = data[0]
        input = input[None, :, :, :]
        preds = 0
        # Reducing to 1 seed for demo purposes
        seeds = [100]  # , 200, 300]
        for seed in seeds:
            model = load_model(seed)
            model.to("cpu")
            output = model(input)
            output = output.cpu().detach().numpy()
            p = np.argmax(output)
            preds += p

        pred = labels[round(preds / len(seeds))]
    return pred


def temp_predict(files):
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(files.getbuffer())
        prediction = predict(tf.name)
    return prediction
