import os
from model.model import Model
from utils.data import SeparatedDataset
from utils.data import DataLoader
from utils.noise_adder import NoiseAdder
from config import Options
from config import logger


def argchk(opt):
    if opt.stage == "tr":
        if opt.log_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.log_dir)
        if opt.ckpt_dir is None:
            raise ValueError
        else:
            logger.info("ckpt_dir:%s", opt.ckpt_dir)
        if opt.tr_data_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.tr_data_dir)
        if opt.cv_data_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.cv_data_dir)
        if opt.tr_noise_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.tr_noise_dir)
        if opt.cv_noise_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.cv_noise_dir)
    elif opt.stage == "tt":
        if opt.tt_data_dir is None:
            raise ValueError
        else:
            logger.info("log_dir:%s", opt.tt_data_dir)

    if opt.audio_log_dir is None:
        raise ValueError
    else:
        logger.info("log_dir: %s", opt.audio_log_dir)

    if opt.gpu_num == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == "__main__":

    # load configrations
    # ==========================================================================
    opt = Options().parse() 
    argchk(opt)
    # ==========================================================================

    if opt.stage == "tr":
        filepatterns = ["*mixture*.wav", "*speech*.wav"]
        tr_data = SeparatedDataset(data_dir_list=[opt.tr_data_dir], 
                                   file_pattern_list=filepatterns)
        cv_data = SeparatedDataset(data_dir_list=[opt.cv_data_dir],
                                   file_pattern_list=filepatterns)
        tr_loader = DataLoader(tr_data, 
                               opt.batch_size * opt.gpu_num,
                               shuffle=True)
        cv_loader = DataLoader(cv_data, opt.batch_size * opt.gpu_num)

        # pre_trained_model_path = "/data/tengxiang/tasnet/log/v0/"
        # ckpt = tf.train.latest_checkpoint(pre_trained_model_path)
        tasnet = Model(opt)
        tasnet.train(tr_loader, cv_loader)

    elif opt.stage == "tt":
        tt_data = SeparatedDataset([opt.tt_data_dir])
        tt_loader = DataLoader(tt_data, opt.batch_size * opt.gpu_num)

        tasnet = Model(opt)
        tasnet.inference(tt_loader, opt.ckpt)
