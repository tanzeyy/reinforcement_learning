import numpy as np
import tensorflow as tf
from utils import Buffer, placeholders, placeholders_from_spaces, mlp_actor_critic, ddpg_mlp_actor_critic, count_vars, get_vars


class AgentBase:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs", self.sess.graph)

    def _build_network(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_action(self, *args, **kwargs):
        raise NotImplementedError


class VPGAgent(AgentBase):

    def _build_network(self):
        state_ph, action_ph = placeholders_from_spaces(
            self.obs_space, self.act_space)
        advantage_ph, return_ph = placeholders(None, None)

        pi, logp, logp_pi, value = mlp_actor_critic(
            state_ph, action_ph, action_space=self.act_space)

        self.all_phs = [state_ph, action_ph, advantage_ph, return_ph, logp_pi]

        self.action_ops = [pi, value, logp_pi]

        # Objectives
        self.pi_loss = -tf.reduce_mean(logp * advantage_ph)
        self.value_loss = tf.reduce_mean((return_ph - value) ** 2)

        # Optimizers
        self.train_pi_opt = tf.train.AdamOptimizer().minimize(self.pi_loss)
        self.train_value_opt = tf.train.AdamOptimizer().minimize(self.value_loss)

    def update(self, inputs, train_v_iters=80):
        feed_dict = {k: v for k, v in zip(self.all_phs, inputs)}
        self.sess.run(self.train_pi_opt, feed_dict=feed_dict)

        for _ in range(train_v_iters):
            self.sess.run(self.train_value_opt, feed_dict=feed_dict)

    def get_action(self, obs):
        act, value, logp_pi = self.sess.run(self.action_ops, feed_dict={
                                   self.all_phs[0]: obs.reshape(1, -1)})
        return act, value, logp_pi


class TRPOAgent(AgentBase):
    pass


class PPOAgent(AgentBase):
    def __init__(self, obs_space, act_space, clip_ratio=0.2):
        self.clip_ratio = clip_ratio
        super().__init__(obs_space, act_space)

    # PPO-clip
    def _build_network(self):
        state_ph, action_ph = placeholders_from_spaces(
            self.obs_space, self.act_space)
        advantage_ph, return_ph, logp_old_ph = placeholders(None, None, None)

        with tf.variable_scope('main'):
            pi, logp, logp_pi, value = mlp_actor_critic(
                state_ph, action_ph, action_space=self.act_space)

        self.all_phs = [state_ph, action_ph,
                        advantage_ph, return_ph, logp_old_ph]

        self.action_ops = [pi, value, logp_pi]

        # PPO Objectives
        ratio = tf.exp(logp - logp_old_ph)
        min_adv = tf.where(advantage_ph > 0, (1 + self.clip_ratio)
                           * advantage_ph, (1 - self.clip_ratio) * advantage_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_ph, min_adv))
        self.value_loss = tf.reduce_mean((return_ph - value) ** 2)
        
        # INFO
        self.approx_kl = tf.reduce_mean(logp_old_ph - logp)
        # self.approx_ent = tf.reduce_mean(-logp)

        # Optimizers
        self.train_pi_opt = tf.train.AdamOptimizer().minimize(self.pi_loss)
        self.train_value_opt = tf.train.AdamOptimizer().minimize(self.value_loss)


    def update(self, inputs, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        feed_dict = {k:v for k, v in zip(self.all_phs, inputs)}
        # self.sess.run(self.train_pi_opt, feed_dict=feed_dict)

        for i in range(train_pi_iters):
            _, kl = self.sess.run([self.train_pi_opt, self.approx_kl], feed_dict=feed_dict)
            kl = np.mean(kl)
            if kl > 1.5 * target_kl:
                print("Early stopping at step {}".format(i))
                break

        for _ in range(train_v_iters):
            self.sess.run(self.train_value_opt, feed_dict=feed_dict)
    
    def get_action(self, obs):
        act, value, logp_pi = self.sess.run(self.action_ops, feed_dict={
                                   self.all_phs[0]: obs.reshape(1, -1)})
        return act, value, logp_pi


class DDPGAgent(AgentBase):
    def __init__(self, obs_space, act_space, hidden_sizes=(300,)):
        self.obs_space = obs_space
        self.act_space = act_space

        # Hyperparameters
        self.gamma = 0.99
        self.noise_scale = 0.2
        self.polyak = 0.995
        self.hidden_sizes = hidden_sizes

        self.sess = tf.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)
        _ = tf.summary.FileWriter("logs", self.sess.graph)


    def _build_network(self):
        obs_dim = self.obs_space.shape[0]
        act_dim = self.act_space.shape[0]
        self.state_ph, self.act_ph, self.next_state_ph, self.rew_ph, self.done_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs
        with tf.variable_scope('main'):
            self.pi, self.q, q_pi = ddpg_mlp_actor_critic(self.state_ph, self.act_ph, action_space=self.act_space, hidden_sizes=self.hidden_sizes)

        # Target networks
        with tf.variable_scope('target'):
            _, _, q_pi_targ = ddpg_mlp_actor_critic(self.next_state_ph, self.act_ph, action_space=self.act_space, hidden_sizes=self.hidden_sizes) # We only need q_pi_targ to compute bellman backup

        var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % var_counts)

        # Bellman target backup
        backup = tf.stop_gradient(self.rew_ph + self.gamma * (1 - self.done_ph) * q_pi_targ)

        # Objectives
        self.q_loss = tf.reduce_mean((self.q - backup) ** 2)
        self.pi_loss = -tf.reduce_mean(q_pi)

        # Optimizers
        self.train_q_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.q_loss, var_list=get_vars('main/q'))
        self.train_pi_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.pi_loss, var_list=get_vars('main/pi'))

        # Polyak averaging for target variables
        self.target_update = tf.group([tf.assign(v_target, self.polyak * v_target + (1 - self.polyak) * v_main) for v_target, v_main in zip(get_vars('target'), get_vars('main'))])
        
        # Init
        self.target_init = tf.group([tf.assign(v_target, v_main) for v_target, v_main in zip(get_vars('target'), get_vars('main'))])

    def get_action(self, obs):
        act_limit = self.act_space.high[0]
        act_dim = self.act_space.shape[0]

        act = self.sess.run(self.pi, feed_dict={self.state_ph:obs.reshape(1,-1)})[0]
        act += self.noise_scale * np.random.randn(act_dim)

        return np.clip(act, -act_limit, act_limit)

    def update(self, batch):
        feed_dict = {
            self.state_ph : batch['obs'], 
            self.act_ph : batch['acts'], 
            self.rew_ph : batch['rews'],
            self.next_state_ph : batch['next_obs'], 
            self.done_ph : batch['done']
        }

        # Q-learning update
        q_loss, _, _ = self.sess.run([self.q_loss, self.q, self.train_q_opt], feed_dict=feed_dict)

        # Policy update
        pi_loss, _, _ = self.sess.run([self.pi_loss, self.target_update, self.train_pi_opt], feed_dict=feed_dict)

        return q_loss, pi_loss
