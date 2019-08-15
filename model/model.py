import os
import time
import numpy as np
import tensorflow as tf
import progressbar as pgb
from . network import TasNet
from . utils import ops
from . utils.utils import AudioChecker
from config import logger


class Model(object):
    def __init__(self, cfgs, new_model=True, model_ckpt=None):
        self.cfgs = cfgs
        self.new_model = new_model
        self.model_ckpt = model_ckpt
        self.gpu_num = cfgs.gpu_num
        self.model = TasNet(
            cfgs.N, cfgs.L, cfgs.B, cfgs.H, cfgs.P, cfgs.X, cfgs.R, cfgs.C,
            causal=cfgs.causal, norm_type=cfgs.norm_type, name="TasNet")
        self.model.build((None, None, 1))
        self.saver = tf.train.Saver()

        self.s_mixed = tf.placeholder(
            dtype=tf.float32,
            shape=[cfgs.batch_size*cfgs.gpu_num, None],
            name="s_mixed")
        self.s_ref = tf.placeholder(
            dtype=tf.float32,
            shape=[cfgs.batch_size*cfgs.gpu_num, None, None],
            name="s_ref")
        self.model_input = tf.expand_dims(self.s_mixed, axis=-1)

        self.towers = self._multi_gpu_run(self.cfgs.batch_size, self.model_input)
        self.s_sep = tf.concat(self.towers["s_sep"], axis=0)
    
        # customal audio checker
        self.mixed_checker = AudioChecker(self.cfgs.audio_log_dir)
        self.ref_spk1_checker = AudioChecker(self.cfgs.audio_log_dir)
        self.sep_spk1_checker = AudioChecker(self.cfgs.audio_log_dir)
        self.noise_checker = AudioChecker(self.cfgs.audio_log_dir)

        self.file_id = "_causal" if self.cfgs.causal else "_noncausal"
        self.file_id += "_{}".format(self.cfgs.norm_type)

        self._sess_config = tf.ConfigProto()
        self._sess_config.gpu_options.allow_growth = True
        self._sess_config.allow_soft_placement = True

    def __getattribute__(self, name):               
        try:
            r = object.__getattribute__(self, name)
        except Exception:
            r = None 
        return r

    def _train_init(self):
        self._add_metric(self.towers)
        self._add_summary()
        self.learning_rate = tf.get_variable(
            name="learning_rate",
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(self.cfgs.lr),
            dtype=tf.float32)
        self.lr_update = tf.maximum(
            tf.assign(self.learning_rate, self.learning_rate / 2,
                      name="update_learning_rate"), 
            self.cfgs.min_lr)
        self.global_step = tf.get_variable(
            name="global_step", shape=[], trainable=False,
            initializer=tf.constant_initializer(0), dtype=tf.int32)
        self.epoch = tf.get_variable(
            name="epoch",
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(0),
            dtype=tf.int32)
        self.min_val_loss = tf.get_variable(
            name="min_val_loss",
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(1e100),
            dtype=tf.float32)
        self.min_val_loss_update = tf.assign(
            self.min_val_loss, self.avg_loss, name="update_min_val_loss")
        self.epoch_update = tf.assign_add(self.epoch, 1, name="update_epoch")

        self.global_opt = self._add_train_op(
            self.towers["loss"], self.learning_rate, self.global_step)

        self.tmp_saver = tf.train.Saver()
        self.sess = tf.Session(config=self._sess_config)
        # self.last_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        self.sess.run(tf.global_variables_initializer())
        if self.model_ckpt is not None:
            self.saver.restore(self.sess, self.model_ckpt)

        self.ckpt_dir = self.cfgs.ckpt_dir 
        self.tmp_ckpt_dir = os.path.join(self.ckpt_dir, "tmp")
        if not os.path.exists(self.tmp_ckpt_dir):
            os.makedirs(self.tmp_ckpt_dir)

    def _add_summary(self):
        # tf.sumamry
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.avg_loss_summary = tf.summary.scalar("loss_avg",
                                                  self.avg_loss,
                                                  collections="epoch")
        self.avg_snr_summary = tf.summary.scalar("si_snr",
                                                 self.si_snr,
                                                 collections="epoch")
        self.summary_op = tf.summary.merge_all()
        self.summary_epoch = tf.summary.merge([self.avg_loss_summary, 
                                               self.avg_snr_summary])
        self.tr_writer = tf.summary.FileWriter(os.path.join(self.cfgs.log_dir, "tr"))
        self.cv_writer = tf.summary.FileWriter(os.path.join(self.cfgs.log_dir, "cv"))
    
    def _add_metric(self, towers):
        towers["snr"] = []
        towers["loss"] = []

        for i in range(self.cfgs.gpu_num):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("gpu_%d" % i):
                    si_snr = ops.si_snr(towers["s_ref"][i], 
                                        towers["s_sep"][i])
                    towers["snr"].append(si_snr)
                    loss = 0 - si_snr
                    towers["loss"].append(loss)

        self.loss = tf.reduce_mean(tf.stack(towers["loss"], axis=0))
        self.si_snr = tf.reduce_mean(tf.stack(towers["snr"], axis=0)) 
        self.avg_loss, self.avg_loss_update = tf.metrics.mean(self.loss)
        self.avg_snr, self.avg_snr_update = tf.metrics.mean(self.si_snr)
        self.avg_update = tf.group(self.avg_loss_update, self.avg_snr_update)
        return towers

    def _multi_gpu_run(self, batch_size, inputs):
        towers = {
            "s_sep": [],
            "s_ref": [],
        }

        for i in range(self.cfgs.gpu_num):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("gpu_%d" % i):
                    s_mixed = inputs[i*batch_size : (i+1)*batch_size]
                    s_ref = self.s_ref[i*batch_size : (i+1)*batch_size]
                    s_ref = s_ref[:, :-self.cfgs.D]
                    s_sep = self.model(s_mixed)[:, self.cfgs.D:]
                    towers["s_sep"].append(s_sep)
                    towers["s_ref"].append(s_ref)
        return towers

    def _add_train_op(self, loss_tower, lr, global_step):
        grads_tower = []
        optimizer = tf.train.AdamOptimizer(lr)
        for i in range(self.cfgs.gpu_num):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("gpu_%d" % i):
                    grads = optimizer.compute_gradients(loss_tower[i])
                    grads_tower.append(grads)
        grads = ops.average_gradients(grads_tower)
        train_op = optimizer.apply_gradients(grads, global_step)
        return train_op

    def _get_feed_dict(self, batch, label=True):
        new_batch = {}
        for data_dict in batch:
            for key in data_dict:
                if new_batch.get(key, None) is None:
                    new_batch[key] = []
                new_batch[key].append(data_dict[key])
        batch_data = list(zip(*new_batch["data"]))
        batch_files = list(zip(*new_batch["filename"]))
        if label:
            feed_dict = {
                self.s_mixed: batch_data[0],
                self.s_ref: batch_data[1],
            }
        else:
            feed_dict = {
                self.s_mixed: batch_data[0],
            }
        return feed_dict, batch_data, batch_files

    def _train(self, dataloader):
        start_time = time.time()
        epoch, lr = self.sess.run([self.epoch, self.learning_rate])
        self.sess.run(tf.local_variables_initializer())

        fetches = {
            "s_mixed": self.s_mixed,
            "s_ref": self.s_ref,
            "s_sep": self.s_sep,
            "loss": self.loss, 
            "avg_loss": self.avg_loss, 
            "snr": self.si_snr,
            "avg_snr": self.avg_snr,
            "log": self.summary_op}

        for idx, batch in enumerate(dataloader):
            feed_dict, batch_data, batch_files = self._get_feed_dict(batch)
            step, *_ = self.sess.run(
                [self.global_step, self.global_opt, self.avg_update],
                feed_dict=feed_dict)

            if step % self.cfgs.log_freq == 0:
                results = self.sess.run(fetches=fetches, feed_dict=feed_dict)
                message = ["TRAIN",
                           "epoch:{:0>2d}".format(epoch), 
                           "loss:{:.2e}".format(results["loss"]), 
                           "avg_loss:{:.2e}".format(results["avg_loss"]),
                           "snr:{:.2f}".format(results["snr"]),
                           "avg_snr:{:.2f}".format(results["avg_snr"]),
                           "lr:{:.2e}".format(lr), 
                           "step:{:d}" .format(step)]
                logger.info(" ".join(message))
                self.tr_writer.add_summary(results["log"], step)

            if step % self.cfgs.save_freq == 0:
                # save temporal audio output files
                mixed_filename = batch_files[0]
                ref_filename = batch_files[1]
                noise_filename = batch_files[-1]
                se_filename = [s.replace("-mixture", self.file_id) 
                               for s in mixed_filename]
                self.mixed_checker.write(
                    mixed_filename, results["s_mixed"], step, stage="tr", epoch=epoch)
                self.ref_spk1_checker.write(
                    ref_filename, results["s_ref"], step, stage="tr", epoch=epoch)
                self.sep_spk1_checker.write(
                    se_filename, results["s_sep"], step, stage="tr", epoch=epoch)

                # save temporal model checkpoints
                self.tmp_saver.save(self.sess,
                                    self.cfgs.ckpt_dir + "tmp/TasNet-se",
                                    global_step=step)
        
        # write epoch-averaged summary
        log_epoch, avg_loss, avg_snr = self.sess.run(
            [self.summary_epoch, self.avg_loss, self.avg_snr], 
            feed_dict=feed_dict)
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        message = ["TRAIN",
                   "epoch:{:0>2d}".format(epoch),
                   "avg_loss:{:.2e}".format(avg_loss),
                   "avg_snr:{:.2f}".format(avg_snr),
                   "period:{:0>2d}h{:0>2d}m{:0>2d}s".format(hh, mm, ss)]
        logger.info(" ".join(message))
        self.tr_writer.add_summary(log_epoch, epoch)

    def _validate(self, dataloader):
        start_time = time.time()
        epoch = self.sess.run(self.epoch)
        self.sess.run(tf.local_variables_initializer())

        fetches = {
            "s_mixed": self.s_mixed,
            "s_ref": self.s_ref,
            "s_sep": self.s_sep,
            "avg_update": self.avg_update}

        for step, batch in enumerate(dataloader):
            feed_dict, batch_data, batch_files = self._get_feed_dict(batch)
            results = self.sess.run(fetches, feed_dict=feed_dict)

            if step % self.cfgs.cv_check_freq == 0:
                mixed_filename = batch_files[0]
                ref_filename = batch_files[1]
                noise_filename = batch_files[-1]
                se_filename = [s.replace("-mixture", self.file_id) 
                               for s in mixed_filename]
                self.mixed_checker.write(
                    mixed_filename, results["s_mixed"], step, stage="cv", epoch=epoch)
                self.ref_spk1_checker.write(
                    ref_filename, results["s_ref"], step, stage="cv", epoch=epoch)
                self.sep_spk1_checker.write(
                    se_filename, results["s_sep"], step, stage="cv", epoch=epoch)
        log_epoch, avg_loss, avg_snr = self.sess.run(
            [self.summary_epoch, self.avg_loss, self.avg_snr],
            feed_dict=feed_dict)
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        message = ["VALIDATE",
                   "epoch:{:0>2d}".format(epoch),
                   "avg_loss:{:.2e}".format(avg_loss),
                   "avg_snr:{:.2f}".format(avg_snr),
                   "period:{:0>2d}h{:0>2d}m{:0>2d}s".format(hh, mm, ss)]
        logger.info(" ".join(message))
        self.cv_writer.add_summary(log_epoch, epoch)
        return avg_loss

    def train(self, tr_loader, cv_loader, resume=False):
        r"""train the model
        
        Args:
            tr_loader (iterable object): an instance of util.data.DataLoader 
            cv_loader (iterable object): an instance of util.data.DataLoader 
            resume (bool, optional): Defaults to False. If True, 
            the training procesing will restore weights from the latest 
            checkpoints and continue to train.
        """
        self._train_init()
        if resume:
            ckpt = tf.train.latest_checkpoint(self.cfgs.ckpt_dir + "/tmp/")
            self.tmp_saver.restore(self.sess, ckpt)
        epoch_counter = self.sess.run(self.epoch)
        val_loss_min = self.sess.run(self.min_val_loss)
        val_loss_counter = 0
        while epoch_counter < self.cfgs.max_epoch:
            self._train(tr_loader)
            val_loss = self._validate(cv_loader)
            if val_loss < val_loss_min or epoch_counter == 0:
                save_model = True
                val_loss_min = self.sess.run(self.min_val_loss_update)
                val_loss_counter = max(0, val_loss_counter - 1)
            else:
                save_model = False
                val_loss_counter += 1
                if val_loss_counter > self.cfgs.epoch_tolerate:
                    self.sess.run(self.lr_update)
            epoch_counter = self.sess.run(self.epoch_update)
            if save_model:
                self.saver.save(self.sess,
                                self.ckpt_dir + "TasNet-se",
                                global_step=epoch_counter)
        self.sess.close()

    def inference(self, dataloader, ckpt=None, evaluate=False):
        start_time = time.time()
        sess = tf.Session(config=self._sess_config)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        if ckpt is not None:
            self.saver.restore(sess, ckpt)
        else:
            self.saver.restore(sess, self.last_ckpt)

        fetches = {"s_sep": self.s_sep}
        if evaluate:
            fetches["avg_update"] = self.avg_update
        
        widgets = [pgb.Percentage(), pgb.Bar('#'), " ", pgb.Timer(), "  ", pgb.ETA()]  
        pbar = pgb.ProgressBar(widgets=widgets, maxval=len(dataloader))
        pbar.start()
        for idx, batch in enumerate(dataloader):
            batch[0]["data"][0] = np.concatenate((batch[0]["data"][0], np.zeros(self.cfgs.D)), axis=-1)
            length = batch[0]["data"][0].shape[0] 
            max_len = 150*16000
            buffer = []
            for i in range(0, length, max_len):
                tmp = batch[0]["data"][0][i:i+max_len+self.cfgs.D]
                feed_dict = {self.s_mixed: [tmp]}
            # feed_dict = self._get_feed_dict(batch_data, label=False)
                results = sess.run(fetches, feed_dict=feed_dict)
                buffer.append(results["s_sep"])
                pbar.update(idx + i / length)
            results["s_sep"] = np.concatenate(buffer, axis=-1)
            mixed_filename = batch[0]["filename"]
            ns_filename = [s.replace("-mixture", self.file_id) for s in mixed_filename]
            self.sep_spk1_checker.write(
                ns_filename, results["s_sep"], idx, stage="tt")
        pbar.finish()
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        message = ["INFERENCE", 
                   "period:{:0>2d}h{:0>2d}m{:0>2d}s".format(hh, mm, ss)]
        logger.info(" ".join(message))

        sess.close()
