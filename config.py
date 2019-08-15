import argparse
import logging

__all__ = ["Options"]

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("TasNet")

# ====================== input settings ========================================
fs = 16000
length = 4 
batch_size = 1
min_sir = 0
max_sir = 20
# ==============================================================================

# ====================== Network settings ======================================
N = 512     # Number of filters in autoencoder
L = 20      # Length of the filters (in samples)
B = 256     # Number of channels in bottleneck 1x1-conv block
H = 512     # Number of channels in convolutional blocks
P = 3       # Kernel size in convolutional blocks
X = 8       # Number of convolutional blocks in each repeat
R = 4       # Numner of repeats
C = 1       # Numebr of speakers
D = int(0.1 * fs)
causal = True
norm_type = "cln"
# ==============================================================================

# ====================== training settings =====================================
lr = 1e-3
min_lr = 1e-7
max_epoch = 100
epoch_tolerate = 2
log_freq = 100 
save_freq = 1000
cv_check_freq = 10
# ==============================================================================


class Options(object):
    """Argument parser"""
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    @staticmethod
    def parse():

        parser = argparse.ArgumentParser()
        parser.add_argument("--stage", default="tr", type=str, 
                            choices=["tt", "tr"],
                            help="train or test the model")
        # ====================== projects settings =============================
        parser.add_argument("--log_dir", default=None, type=str,
                            help="directory to save training log")
        parser.add_argument("--ckpt_dir", default=None, type=str,
                            help="directory to save checkpoints")
        parser.add_argument("--tr_data_dir", default=None, type=str,
                            help="path to training set")
        parser.add_argument("--cv_data_dir", default=None, type=str,
                            help="path to validation set")
        parser.add_argument("--tr_noise_dir", default=None, type=str,
                            help="path to training noise set")
        parser.add_argument("--cv_noise_dir", default=None, type=str,
                            help="path to validation noise set")
        parser.add_argument("--tt_data_dir", default=None, type=str,
                            help="path to test set")
        parser.add_argument("--audio_log_dir", default=None, type=str,
                            help="directory to save audio files")
        parser.add_argument("--ckpt", default=None, type=str,
                            help="path to checkpoint")
        
        # ====================== input settings ================================
        parser.add_argument("-fs", default=fs, help="audio sampling frequency")
        parser.add_argument("--length", type=float, default=length,
                            help="length of the input audio file (in seconds)")
        parser.add_argument("--batch_size", type=int, default=batch_size,
                            help="length of the input audio file (in seconds)")
        # ====================== network settings ================================
        parser.add_argument("-N", type=int, default=N,
                            help="number of filters in encoder")
        parser.add_argument("-L", type=int, default=L,
                            help="length of the filters in encoder (in samples)")
        parser.add_argument("-B", type=int, default=B,
                            help="number of channels in bottleneck layer")
        parser.add_argument("-H", type=int, default=H,
                            help="number of channels in convolutional blocks")
        parser.add_argument("-P", type=int, default=P,
                            help="kernel size in convolutional blocks")
        parser.add_argument("-X", type=int, default=X,
                            help="number of conv-block in each repeat")
        parser.add_argument("-R", type=int, default=R,
                            help="number of repeats")
        parser.add_argument("-C", type=int, default=C,
                            help="number of speakers")
        parser.add_argument("--causal", type=bool, default=causal,
                            help="processing causally")
        parser.add_argument("--norm_type", type=str, default=norm_type, 
                            choices=["cln", "gln"],
                            help="type of normalization")

        # ====================== training settings ================================
        parser.add_argument("--lr", type=float, default=lr,
                            help="initial learning rate")
        parser.add_argument("--min_lr", type=float, default=min_lr,
                            help="minimal learning rate")
        parser.add_argument("--max_epoch", type=int, default=max_epoch,
                            help="maximum epochs to train")
        parser.add_argument("--epoch_tolerate", type=int, default=epoch_tolerate,
                            help="maximum epochs to train")
        parser.add_argument("--log_freq", type=int, default=log_freq,
                            help="log printing frequency")
        parser.add_argument("--save_freq", type=int, default=save_freq,
                            help="model saving frequency")
        parser.add_argument("--cv_check_freq", type=int, default=cv_check_freq,
                            help="printing and saving frequency on validation stage")
        
        parser.add_argument("--gpu_num", type=int, default=1,
                            help="number of gpu to use")
        opt = parser.parse_args()

        opt.length = int(opt.length * opt.fs)
        return opt 
