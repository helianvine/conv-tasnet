import os
from model.model import Model
from utils.data import SeparatedDataset
from utils.data import DataLoader
from config import Options

CMD_DATA1 = "/data/zhangyi/Noisy_dataset/CHJ14_cmd-CHJ/CHJ14_cmd-CHJ-2/CHJ14_cmd-CHJ/"
CMD_DATA2 = "/data/zhangyi/Noisy_dataset/CHJ14_cmd-CHJ/CHJ14_cmd-CHJ-3/CHJ14_cmd-CHJ/hp/"
AISHELL_DATA = "/data/tengxiang/dataset/aishell-chj-hp/"
EXTRA_DATA = "/data/tengxiang/dataset/wenzhifan/4CH/"
HISF_DATA = "/data/tengxiang/dataset/hisf/cmd_input/wav/"
LOG_HOME = "/data/tengxiang/aishell_hp/"

# tt_data_dir = os.path.join(AISHELL_DATA, "cv")
tt_data_dir = EXTRA_DATA
audio_log_dir = os.path.join(EXTRA_DATA, "../v1")
ckpt = "/home/teng.xiang/projects/model-ckpt/aishell_hp-sir_005_020/TasNet-se-19"
# ckpt = "/data/tengxiang/aishell_hp/v0/ckpt/TasNet-se-9"
# ckpt = "/home/teng.xiang/projects/model-ckpt/aishell_time-delay_cln/TasNet-se-17"

if __name__ == "__main__":
    opt = Options().parse() 
    if opt.gpu_num == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if opt.tt_data_dir is None:
        opt.tt_data_dir = tt_data_dir
    if opt.audio_log_dir is None:
        opt.audio_log_dir = audio_log_dir

    tt_data = SeparatedDataset([opt.tt_data_dir])
    tt_loader = DataLoader(tt_data, opt.batch_size * opt.gpu_num)

    tasnet = Model(opt)
    tasnet.inference(tt_loader, ckpt)
