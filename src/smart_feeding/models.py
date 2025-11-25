import torch
import torch.nn as nn
import torch.nn.functional as F
from .ultils import do_mixup
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def init_layer(layer):
    """
    Initialize weights for a Linear or Convolutional layer using Xavier uniform.
    Also zero-initialize biases if present.
    """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """
    Initialize a BatchNorm layer's parameters to sensible defaults.
    """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """Forward pass for a ConvBlock5x5 followed by pooling.

        pool_type: 'avg', 'max' or 'avg+max'.
        """
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x
    
# --- CNN6 model ---
class Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num):

        """
        Cnn6 audio classification model.

        Parameters follow the training config: sample_rate, window_size, hop_size,
        mel_bins, fmin, fmax, classes_num.
        """
        super(Cnn6, self).__init__()
        self.mel_bins = mel_bins

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Extraction Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Extraction Logmel
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)


        self.spec_augmenter = SpecAugmentation(time_drop_width=32, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(self.mel_bins) 

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Forward pass.

        Input expected as waveform tensor (batch_size, samples).
        If `mixup_lambda` is provided during training, mixup will be applied.
        Returns a dict with 'clipwise_output' and 'embedding'.
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # Transpose và BatchNorm
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup trên spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Các khối Conv
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Pooling
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # Fully Connected layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        # Output layer (Logits when training, Softmax when eval)
        if self.training:
            clipwise_output = self.fc_audioset(x)
        else:
            clipwise_output = torch.softmax(self.fc_audioset(x), dim=-1)

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict