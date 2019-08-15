
import tensorflow as tf
from itertools import permutations
from . utils import EPS


def assign_moving_average(var, val, decay):
    with tf.name_scope("assign_moving_average"):
        var1 = tf.assign(var, var * decay + val * (1 - decay))
    return var1


def si_snr(s_sep, s_ref):
    """scale-invariant signal-to-noise ratio
    
    Args:
        s_sep (Tensor): degradated signal with shape [batch_size, speakers, legnth]
        s_ref (Tensor): reference signal with shape [batch_size, speakers, length]
    """

    with tf.name_scope("SI_SNR"):
        with tf.name_scope("zero_mean"):
            s_sep_mean = tf.reduce_mean(s_sep, axis=-1, keep_dims=True)
            s_ref_mean = tf.reduce_mean(s_ref, axis=-1, keep_dims=True)
            s_sep -= s_sep_mean  # [batch_size, C(speakers), length]
            s_ref -= s_ref_mean  # [batch_size, C(speakers), length]
        s_dot = tf.reduce_sum(s_ref * s_sep, axis=-1,
                              keep_dims=True)  # [batch_size, C(speakers), 1]
        p_ref = tf.reduce_sum(s_ref**2, axis=-1,
                              keep_dims=True)  # [batch_size, C(speakers), 1]
        s_target = s_dot * s_ref / (p_ref + EPS
                                    )  # [batch_size, C(speakers), length]
        e_noise = s_sep - s_target  # [batch_size, C(speakers), length]
        s_target_norm = tf.reduce_sum(s_target**2,
                                      axis=-1)  # [batch_size, C(speakers)]
        e_noise_norm = tf.reduce_sum(e_noise**2,
                                     axis=-1)  # [batch_size, C(speakers)]
        si_snr = 10 * tf.log(s_target_norm /
                             (e_noise_norm + EPS))  # [batch_size, C(speakers)]
        si_snr = tf.reduce_mean(si_snr) / tf.log(10.0)
    return si_snr


def pit_si_snr(s_sep, s_ref, speakers):
    """permutation-invariant scale-invariant signal-to-noise ratio
    
    Args:
        s_sep (Tensor): degradated signal with shape [batch_size, speakers, legnth]
        s_ref (Tensor): reference signal with shape [batch_size, speakers, length]
    """
    with tf.name_scope("pit_snr"):
        batch_size = tf.shape(s_ref)[0]

        with tf.name_scope("zero_mean"):
            s_sep_mean = tf.reduce_mean(s_sep, axis=-1, keep_dims=True)
            s_ref_mean = tf.reduce_mean(s_ref, axis=-1, keep_dims=True)
            s_sep -= s_sep_mean  # [batch_size, C(speakers), length]
            s_ref -= s_ref_mean  # [batch_size, C(speakers), length]

        s_sep = tf.expand_dims(s_sep, axis=1)  # [batch_size, 1, C, length]
        s_ref = tf.expand_dims(s_ref, axis=2)  # [batch_size, C, 1, length]
        pair_wise_dot = tf.reduce_sum(s_ref * s_sep, axis=-1,
                                      keep_dims=True)  # [batch_size, C, C, 1]
        p_ref = tf.reduce_sum(s_ref**2, axis=-1,
                              keep_dims=True)  # [batch_size, C, 1, 1]
        s_target = pair_wise_dot * s_ref / (p_ref + EPS
                                            )  # [batch_size, C, C, length]
        e_noise = s_sep - s_target  # [batch_size, C, C, length]
        s_target_norm = tf.reduce_sum(s_target**2, axis=-1)
        e_noise_norm = tf.reduce_sum(e_noise**2, axis=-1)
        pair_wise_snr = 10 * tf.log(s_target_norm /
                                    (e_noise_norm + EPS))  # [batch_size, C, C]

        perms = tf.constant(list(permutations(range(speakers))))  # [C!, C]
        perms_one_hot = tf.one_hot(perms, depth=speakers)  # [C!, C, C]
        snr_set = tf.einsum("bij,pij->bp", pair_wise_snr,
                            perms_one_hot)  # [batch_size, C!]

        max_snr_index = tf.arg_max(snr_set, 1)
        batch_idx = get_batch_idx(batch_size, max_snr_index)
        batch_perm = tf.gather_nd(perms, batch_idx)
        speakers = tf.constant(speakers, dtype=tf.float32)
        max_snr = tf.reduce_max(snr_set, 1) / (speakers * tf.log(10.0))
    return tf.reduce_mean(max_snr), batch_perm


def get_batch_idx(batch_size, idx):
    batch_list = tf.range(batch_size)
    batch_idx = tf.stack([batch_list, tf.cast(idx, tf.int32)], axis=1)
    return batch_idx


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
