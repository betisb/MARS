import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import restore_tf_graph
import os.path as osp
from HPC_Simulation import *


def policy_loader(model_path, itr='last'):
    if itr == 'last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save' + itr))
    pi = model['pi']
    v = model['v']
    get_probs = lambda x, y: sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * TASK_FEATURES),
                                                     model['mask']: y.reshape(-1, MAX_QUEUE_SIZE)})
    get_v = lambda x: sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * TASK_FEATURES)})
    return get_probs, get_v

def mlp3(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, TASK_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)

def mlp(x, act_dim):
    x = tf.reshape(x, shape=[-1, TASK_SEQUENCE_SIZE, TASK_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)


def mlp2(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, TASK_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x


def attention(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, TASK_FEATURES])
    q = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    k = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    v = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    score = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
    score = tf.nn.softmax(score, -1)
    attn = tf.reshape(score, (-1, MAX_QUEUE_SIZE, MAX_QUEUE_SIZE))
    x = tf.matmul(attn, v)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)

    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x


def lenet(x_ph, act_dim):
    m = int(np.sqrt(MAX_QUEUE_SIZE))
    x = tf.reshape(x_ph, shape=[-1, m, m, TASK_FEATURES])
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[1, 1], strides=1)
    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[1, 1], strides=1)
    x = tf.layers.max_pooling2d(x, [2, 2], 2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=64)

    return tf.layers.dense(
        inputs=x,
        units=act_dim,
        activation=None
    )

def categorical_policy(x, a, mask, action_space, attn):
    act_dim = action_space.n
    if attn:
        output_layer = attention(x, act_dim)
    else:
        output_layer = lenet(x, act_dim)
    output_layer = output_layer + (mask - 1) * 1000000
    logp_all = tf.nn.log_softmax(output_layer)

    pi = tf.squeeze(tf.multinomial(output_layer, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, output_layer

def actor_critic(x, a, mask, action_space=None, attn=False):
    with tf.variable_scope('pi'):
        pi, logp, logp_pi, out = categorical_policy(x, a, mask, action_space, attn)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp3(x, 1), axis=1)
    return pi, logp, logp_pi, v, out


class MARSBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size * 100
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.cobs_buf = None
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, cobs, act, mask, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        actual_adv_buf = np.array(self.adv_buf, dtype=np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        return [self.obs_buf[:actual_size], self.act_buf[:actual_size], self.mask_buf[:actual_size], actual_adv_buf,
                self.ret_buf[:actual_size], self.logp_buf[:actual_size]]

def mars(workload_file, model_path, ac_kwargs=dict(), seed=0,
        traj_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, pre_trained=0, trained_model=None, attn=False,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = HPC_Environment(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn
    buf = MARSBuffer(obs_dim, act_dim, traj_per_epoch * TASK_SEQUENCE_SIZE, gamma, lam)

    if pre_trained:
        sess = tf.Session()
        model = restore_tf_graph(sess, trained_model)
        logger.log('loading model')
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
        x_ph = model['x']
        a_ph = model['a']
        mask_ph = model['mask']
        adv_ph = model['adv']
        ret_ph = model['ret']
        logp_old_ph = model['logp_old_ph']
        pi = model['pi']
        v = model['v']
        out = model['out']
        logp = model['logp']
        logp_pi = model['logp_pi']
        pi_loss = model['pi_loss']
        v_loss = model['v_loss']
        approx_ent = model['approx_ent']
        approx_kl = model['approx_kl']
        clipfrac = model['clipfrac']
        clipped = model['clipped']
        train_pi = tf.get_collection("train_pi")[0]
        train_v = tf.get_collection("train_v")[0]
        all_phs = [x_ph, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
        get_action_ops = [pi, v, logp_pi, out]

    else:
        if(buf < 512):
            x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
            mask_ph = placeholder(MAX_QUEUE_SIZE)
            adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)
            pi, logp, logp_pi, v, out = actor_critic(x_ph, a_ph, mask_ph, **ac_kwargs)
            all_phs = [x_ph, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
            get_action_ops = [pi, v, logp_pi, out]
            var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
            logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
            ratio = tf.exp(logp - logp_old_ph)
            min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
            v_loss = tf.reduce_mean((ret_ph - v) ** 2)
            approx_kl = tf.reduce_mean(logp_old_ph - logp)
            approx_ent = tf.reduce_mean(-logp)
            clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
            clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
            train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
            train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            tf.add_to_collection("train_pi", train_pi)
            tf.add_to_collection("train_v", train_v)
        else:
            x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
            mask_ph = placeholder(MAX_QUEUE_SIZE)
            adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)
            pi, logp, logp_pi, v, out = actor_critic(x_ph, a_ph, mask_ph, **ac_kwargs)
            all_phs = [x_ph, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
            get_action_ops = [pi, v, logp_pi, out]
            var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
            logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
            ratio = tf.exp(logp - logp_old_ph)
            min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
            v_loss = tf.reduce_mean((ret_ph - v) ** 2)
            approx_kl = tf.reduce_mean(logp_old_ph - logp)
            approx_ent = tf.reduce_mean(-logp)
            clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
            clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
            train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
            train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            tf.add_to_collection("train_pi", train_pi)
            tf.add_to_collection("train_v", train_v)
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph, 'adv': adv_ph, 'mask': mask_ph, 'ret': ret_ph,
                                        'logp_old_ph': logp_old_ph},
                          outputs={'pi': pi, 'v': v, 'out': out, 'pi_loss': pi_loss, 'logp': logp, 'logp_pi': logp_pi,
                                   'v_loss': v_loss, 'approx_ent': approx_ent, 'approx_kl': approx_kl,
                                   'clipped': clipped, 'clipfrac': clipfrac})
    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Max reached at step %d ' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
    start_time = time.time()
    num_total = 0
    for epoch in range(epochs):
        t = 0
        while True:
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * TASK_FEATURES, TASK_FEATURES):
                if all(o[i:i + TASK_FEATURES] == [0] + [1] * (TASK_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i:i + TASK_FEATURES] == [1] * TASK_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)

            a, v_t, logp_t, output = sess.run(get_action_ops,
                                              feed_dict={x_ph: o.reshape(1, -1), mask_ph: np.array(lst).reshape(1, -1)})

            num_total += 1
            buf.store(o, None, a, np.array(lst), r, v_t, logp_t)
            logger.store(VVals=v_t)
            o, r, d, r2, sjf_t, f1_t = env.step(a[0])
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t
            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)
                [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
                if t >= traj_per_epoch:
                    break
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
        update()
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * traj_per_epoch * TASK_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('ShowRet', average_only=True)
        logger.log_tabular('SJF', average_only=True)
        logger.log_tabular('F1', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str,
                        default='./Dataset/synthetic_small.swf')
    parser.add_argument('--model', type=str, default='./Dataset/synthetic_small.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='mars')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./output/result_00')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './output/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    if args.pre_trained:
        model_file = os.path.join(current_dir, args.trained_model)
        mars(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
            logger_kwargs=logger_kwargs, pre_trained=1, trained_model=os.path.join(model_file, "simple_save"),
            attn=args.attn,
            shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, score_type=args.score_type,
            batch_job_slice=args.batch_job_slice)
    else:
        mars(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
            logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn, shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)
