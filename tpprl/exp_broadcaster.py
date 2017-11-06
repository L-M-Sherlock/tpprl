import numpy as np
import redqueen.opt_model as OM
import redqueen.utils as RU
import tensorflow as tf
import decorated_options as Deco
from exp_sampler import ExpCDFSampler

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


class ExpRecurrentBroadcaster(OM.Broadcaster):
    """This is a broadcaster which follows the intensity function as defined by
    RMTPP paper and updates the hidden state upon receiving each event.

    TODO: The problem is that calculation of the gradient and the loss/LL
    becomes too complicated with numerical stability issues very quickly. Need
    to implement adaptive scaling to handle that issue.

    Also, this embeds the event history implicitly and the state function does
    not explicitly model the loss function J(.) faithfully. This is an issue
    with the theory.
    """

    @Deco.optioned()
    def __init__(self, src_id, seed, trainer, t_min=0):
        super(ExpRecurrentBroadcaster, self).__init__(src_id, seed)
        self.init = False

        self.trainer = trainer

        params = Deco.Options(**self.trainer.sess.run({
            'Wm': trainer.tf_Wm,
            'Wh': trainer.tf_Wh,
            'Bh': trainer.tf_Bh,
            'Wt': trainer.tf_Wt,
            'Wr': trainer.tf_Wr,

            'wt': trainer.tf_wt,
            'vt': trainer.tf_vt,
            'bt': trainer.tf_bt,
            'init_h': trainer.tf_h
        }))

        self.cur_h = params.init_h

        self.exp_sampler = ExpCDFSampler(_opts=params,
                                         t_min=t_min,
                                         seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here.
        r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)

        feed_dict = {
            self.trainer.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
            self.trainer.tf_t_delta: np.asarray([time_delta]).reshape(-1),
            self.trainer.tf_h: self.cur_h,
            self.trainer.tf_rank: np.asarray([np.mean(r_t)]).reshape(-1)
        }
        return self.trainer.sess.run(self.trainer.tf_h_next,
                                     feed_dict=feed_dict)

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            # Nothing special to do for the first event.

        self.state.apply_event(event)

        if event is None:
            # This is the first event. Post immediately to join the party?
            # Or hold off?
            return self.exp_sampler.generate_sample()
        else:
            self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
            next_post_time = self.exp_sampler.register_event(
                event.cur_time,
                self.cur_h,
                own_event=event.src_id == self.src_id
            )
            next_delta = next_post_time - self.last_self_event_time
            # print(next_delta)
            assert next_delta >= 0
            return next_delta


OM.SimOpts.registerSource('ExpRecurrentBroadcaster', ExpRecurrentBroadcaster)


class ExpRecurrentTrainer:

    @Deco.optioned()
    def __init__(self, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, init_h, sess, sim_opts,
                 scope=None, t_min=0, batch_size=16, max_events=100):
        """Initialize the trainer with the policy parameters."""
        self.src_embed_map = {x.src_id: idx + 1
                              for idx, x in enumerate(sim_opts.create_other_sources())}
        self.src_embed_map[sim_opts.src_id] = 0

        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.q = 10.0
        self.batch_size = batch_size
        self.max_events = max_events
        self.num_hidden_states = init_h.shape[0]

        init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))

        self.scope = scope or type(self).__name__

        # TODO: Create all these variables on the CPU and the training vars on the GPU
        # by using tf.device explicitly?
        with tf.variable_scope(self.scope):
            with tf.variable_scope("hidden_state"):
                self.tf_Wm = tf.get_variable(name="Wm", shape=Wm.shape,
                                             initializer=tf.constant_initializer(Wm))
                self.tf_Wh = tf.get_variable(name="Wh", shape=Wh.shape,
                                             initializer=tf.constant_initializer(Wh))
                self.tf_Wt = tf.get_variable(name="Wt", shape=Wt.shape,
                                             initializer=tf.constant_initializer(Wt))
                self.tf_Wr = tf.get_variable(name="Wr", shape=Wr.shape,
                                             initializer=tf.constant_initializer(Wr))
                self.tf_Bh = tf.get_variable(name="Bh", shape=Bh.shape,
                                             initializer=tf.constant_initializer(Bh))
                self.tf_h = tf.get_variable(name="h", shape=(self.num_hidden_states, 1),
                                            initializer=tf.constant_initializer(init_h))
                self.tf_b_idx = tf.placeholder(name="b_idx", shape=1, dtype=tf.int32)
                self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=self.tf_dtype)
                self.tf_rank = tf.placeholder(name="rank", shape=1, dtype=self.tf_dtype)

                self.tf_h_next = tf.nn.tanh(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name="b_embed")
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) +
                    self.tf_Wr * self.tf_rank +
                    self.tf_Wt * self.tf_t_delta +
                    self.tf_Bh,
                    name="h_next"
                )

                # self.tf_h_next = tf.nn.

            with tf.variable_scope("output"):
                self.tf_bt = tf.get_variable(name="bt", shape=bt.shape,
                                             initializer=tf.constant_initializer(bt))
                self.tf_vt = tf.get_variable(name="vt", shape=vt.shape,
                                             initializer=tf.constant_initializer(vt))
                self.tf_wt = tf.get_variable(name="wt", shape=wt.shape,
                                             initializer=tf.constant_initializer(wt))
                # self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=self.tf_dtype)
                # self.tf_u_t = tf.exp(
                #     tf.tensordot(self.tf_vt, self.tf_h, axes=1) +
                #     self.tf_t_delta * self.tf_wt +
                #     self.tf_bt,
                #     name="u_t"
                # )

            # Create a large dynamic_rnn kind of network which can calculate
            # the gradients for a given given batch of simulations.
            with tf.variable_scope("training"):
                self.tf_batch_rewards = tf.placeholder(name="rewards",
                                                       shape=(batch_size, 1),
                                                       dtype=self.tf_dtype)
                self.tf_batch_t_deltas = tf.placeholder(name="t_deltas",
                                                        shape=(batch_size, max_events),
                                                        dtype=self.tf_dtype)
                self.tf_batch_b_idxes = tf.placeholder(name="b_idxes",
                                                       shape=(batch_size, max_events),
                                                       dtype=tf.int32)
                self.tf_batch_ranks = tf.placeholder(name="ranks",
                                                     shape=(batch_size, max_events),
                                                     dtype=self.tf_dtype)
                self.tf_batch_seq_len = tf.placeholder(name="seq_len",
                                                       shape=(batch_size, 1),
                                                       dtype=tf.int32)
                self.tf_batch_last_interval = tf.placeholder(name="last_interval",
                                                             shape=batch_size,
                                                             dtype=self.tf_dtype)

                self.tf_batch_init_h = tf_batch_h_t = tf.zeros(name="init_h",
                                                               shape=(batch_size, self.num_hidden_states),
                                                               dtype=self.tf_dtype)

                self.h_states = []
                self.LL_log_terms = []
                self.LL_int_terms = []
                self.loss_terms = []

                # self.LL = tf.zeros(name="log_likelihood", dtype=self.tf_dtype, shape=(batch_size))
                # self.loss = tf.zeros(name="loss", dtype=self.tf_dtype, shape=(batch_size))

                t_0 = tf.zeros(name="event_time", shape=batch_size, dtype=self.tf_dtype)

                def batch_u_theta(batch_t_deltas):
                    return tf.exp(
                        tf.matmul(tf_batch_h_t, self.tf_vt) +
                        self.tf_wt * tf.expand_dims(batch_t_deltas, 1) +
                        self.tf_bt
                    )

                # TODO: Convert this to a tf.while_loop, perhaps.
                # The performance benefit is debatable.
                for evt_idx in range(max_events):
                    # Perhaps this can be melded into the old definition of tf_h_next
                    # above by using the batch size dimension as None?
                    # TODO: Investigate
                    tf_batch_h_t = tf.where(
                        tf.tile(evt_idx <= self.tf_batch_seq_len, [1, self.num_hidden_states]),
                        tf.nn.tanh(
                            tf.nn.embedding_lookup(self.tf_Wm,
                                                   self.tf_batch_b_idxes[:, evt_idx]) +
                            tf.matmul(tf_batch_h_t, self.tf_Wh, transpose_b=True) +
                            tf.matmul(tf.expand_dims(self.tf_batch_ranks[:, evt_idx], 1),
                                      self.tf_Wr, transpose_b=True) +
                            tf.matmul(tf.expand_dims(self.tf_batch_t_deltas[:, evt_idx], 1),
                                      self.tf_Wt, transpose_b=True) +
                            tf.tile(tf.transpose(self.tf_Bh), [batch_size, 1])
                        ),
                        tf.zeros(dtype=self.tf_dtype, shape=(batch_size, self.num_hidden_states))
                        # The gradient of a constant w.r.t. a variable is None or 0
                    )
                    tf_batch_u_theta = tf.where(
                        evt_idx <= self.tf_batch_seq_len,
                        batch_u_theta(self.tf_batch_t_deltas[:, evt_idx]),
                        tf.zeros(dtype=self.tf_dtype, shape=(batch_size, 1))
                    )

                    self.h_states.append(tf_batch_h_t)
                    self.LL_log_terms.append(tf.where(
                        tf.squeeze(evt_idx <= self.tf_batch_seq_len),
                        tf.where(
                            tf.equal(self.tf_batch_b_idxes[:, evt_idx], sim_opts.src_id),
                            tf.squeeze(tf.log(tf_batch_u_theta)),
                            tf.zeros(dtype=self.tf_dtype, shape=batch_size)),
                        tf.zeros(dtype=self.tf_dtype, shape=batch_size)))

                    self.LL_int_terms.append(tf.where(
                        tf.squeeze(evt_idx <= self.tf_batch_seq_len),
                        - (1 / self.tf_wt) * tf.squeeze(
                            batch_u_theta(t_0) -
                            tf_batch_u_theta
                        ),
                        tf.zeros(dtype=self.tf_dtype, shape=batch_size)))

                    self.loss_terms.append(tf.where(
                        tf.squeeze(evt_idx <= self.tf_batch_seq_len),
                        -(1 / (2 * self.tf_wt)) * tf.squeeze(
                            tf.square(batch_u_theta(t_0)) -
                            tf.square(tf_batch_u_theta)
                        ),
                        tf.zeros(dtype=self.tf_dtype, shape=(batch_size))))

        self.LL = tf.add_n(self.LL_log_terms) + tf.add_n(self.LL_log_terms)
        self.loss = tf.add_n(self.loss_terms)

        # Here, outside the loop, add the survival term for the batch to
        # both the loss and to the LL.
        self.LL += (1 / self.tf_wt) * tf.squeeze(
            batch_u_theta(t_0) - batch_u_theta(self.tf_batch_last_interval)
        )
        self.loss += - (1 / (2 * self.tf_wt)) * tf.squeeze(
            tf.square(batch_u_theta(t_0)) -
            tf.square(batch_u_theta(self.tf_batch_last_interval))
        )
        self.loss *= self.q / 2

        self.all_tf_vars = [self.tf_Wh, self.tf_Wm, self.tf_Wt, self.tf_Bh,
                            self.tf_Wr,
                            self.tf_bt, self.tf_vt, self.tf_wt]

        # The gradients are added over the batch if made into a single call.
        # TODO: Perhaps there is a faster way of calculating these gradients?
        self.LL_grads = {x: [tf.gradients(y, x)
                             for y in tf.split(self.LL, self.batch_size)]
                         for x in self.all_tf_vars}
        self.loss_grads = {x: [tf.gradients(y, x)
                               for y in tf.split(self.loss, self.batch_size)]
                           for x in self.all_tf_vars}

        # Attempt to calculate the gradient within Tensorflow for the entire
        # batch, without moving to the CPU.
        self.tower_gradients = [
            [(
                # TODO: This looks horribly inefficient and should be replaced by
                # matrix multiplication soon.
                (tf.gather(self.tf_batch_rewards, idx) + tf.gather(self.loss, idx)) * self.LL_grads[x][idx][0] +
                self.loss_grads[x][idx][0],
                x
             )
             for x in self.all_tf_vars]
            for idx in range(self.batch_size)
        ]
        self.avg_gradient = average_gradients(self.tower_gradients)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.sgd_op = self.opt.apply_gradients(self.avg_gradient)

        self.sim_opts = sim_opts
        self.src_id = sim_opts.src_id
        self.sess = sess

    def initialize(self, finalize=True):
        """Initialize the graph."""
        self.sess.run(tf.global_variables_initializer())
        if finalize:
            # No more nodes will be added to the graph beyond this point.
            # Recommended way to prevent memory leaks afterwards, esp. if the
            # session will be used in a multi-threaded manner.
            # https://stackoverflow.com/questions/38694111/
            self.sess.graph.finalize()

    def _create_exp_broadcaster(self, seed):
        """Create a new exp_broadcaster with the current params."""
        return ExpRecurrentBroadcaster(src_id=self.src_id, seed=seed, trainer=self)

    def run_sim(self, seed):
        """Run one simulation and return the dataframe.
        Will be thread-safe and can be called multiple times."""
        run_sim_opts = self.sim_opts.update({})
        exp_b = self._create_exp_broadcaster(seed=seed * 3)

        mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)
        mgr.run_dynamic()
        return mgr.get_state().get_dataframe()

    def reward_fn(self, df):
        """Calculate the reward for a given trajectory."""
        rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
        rank_dt = np.diff(np.concatenate([rank_in_tau.index.values,
                                          [self.sim_opts.end_time]]))
        return np.sum((rank_in_tau ** 2) * rank_dt)

    def get_feed_dict(self, batch_df):
        """Produce a feed_dict for the given batch."""
        assert all(len(df.sink_id.unique()) == 1 for df in batch_df), "Can only handle one sink at the moment."
        assert len(batch_df) == self.batch_size, "The batch should consist of {} simulations, not {}.".format(self.batch_size, len(batch_df))

        full_shape = (self.batch_size, self.max_events)

        batch_rewards = np.asarray([self.reward_fn(x) for x in batch_df])[:, np.newaxis]
        batch_t_deltas = np.zeros(shape=full_shape, dtype=float)

        batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
        batch_ranks = np.zeros(shape=full_shape, dtype=float)
        batch_seq_len = np.asarray([np.minimum(x.shape[0], self.max_events) for x in batch_df], dtype=int)[:, np.newaxis]
        batch_init_h = np.zeros(shape=(self.batch_size, self.num_hidden_states), dtype=float)

        batch_last_interval = np.zeros(shape=self.batch_size, dtype=float)

        for idx, df in enumerate(batch_df):
            # They are sorted by time already.
            batch_len = int(batch_seq_len[idx])
            rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
            batch_ranks[idx, 0:batch_len] = rank_in_tau.values[0:batch_len]
            batch_b_idxes[idx, 0:batch_len] = df.src_id.map(self.src_embed_map).values[0:batch_len]
            batch_t_deltas[idx, 0:batch_len] = df.time_delta.values[0:batch_len]
            if batch_len == df.shape[0]:
                # This batch has consumed all the events
                batch_last_interval[idx] = self.sim_opts.end_time - df.t.iloc[-1]
            else:
                batch_last_interval[idx] = df.time_delta[batch_len]

        return {
            self.tf_batch_b_idxes: batch_b_idxes,
            self.tf_batch_rewards: batch_rewards,
            self.tf_batch_seq_len: batch_seq_len,
            self.tf_batch_t_deltas: batch_t_deltas,
            self.tf_batch_ranks: batch_ranks,
            self.tf_batch_init_h: batch_init_h,
            self.tf_batch_last_interval: batch_last_interval,
        }

    def get_batch_grad(self, batch):
        """Returns the true gradient, given a feed dictionary generated by get_feed_dict."""
        feed_dict = self.get_feed_dict(batch)
        batch_rewards = [self.reward_fn(x) for x in batch]

        # The gradients are already summed over the batch dimension.
        LL_grads, losses, loss_grads = self.sess.run([self.LL_grads, self.loss, self.loss_grads],
                                                     feed_dict=feed_dict)

        true_grads = []
        for batch_idx in range(len(batch)):
            reward = batch_rewards[batch_idx]
            loss = losses[batch_idx]
            # TODO: Is there a better way of working with IndexesSlicedValue
            # then converting it to a dense numpy array? Probably not.
            batch_grad = {}
            for x in self.all_tf_vars:
                LL_grad = LL_grads[x][batch_idx][0]

                if hasattr(LL_grad, 'dense_shape'):
                    np_LL_grad = np.zeros(LL_grad.dense_shape, dtype=self.np_dtype)
                    np_LL_grad[LL_grad.indices] = LL_grad.values
                else:
                    np_LL_grad = LL_grad

                loss_grad = loss_grads[x][batch_idx][0]

                if hasattr(loss_grad, 'dense_shape'):
                    np_loss_grad = np.zeros(loss_grad.dense_shape)
                    np_loss_grad[loss_grad.indices] = loss_grad.values
                else:
                    np_loss_grad = loss_grad

                batch_grad[x] = (reward + loss) * np_LL_grad + np_loss_grad

            true_grads.append(batch_grad)

        return true_grads

    def train(self, sim_batch):
        """Run one SGD op given a batch of simulation."""



###################################
# This is an attempt to remain more faithful to the theory proposed in the paper.
# Here, we will do model predictive control of sorts by determining what is the
# current intensity of each broadcaster and the predicted mean next event time.
#  - Issue: the mean may be infinity for certain kinds of broadcasters.
#           - Maybe consider the median instead, which may also be infinite?
#           - TODO: Verify because ∫tf(t)dt does not look like it will go to
#             infinity if f(t) decays too quickly?
#           - Model infinity by using an indicator variable (more elegant)
#           - This is especially important if we use the RMTPP formulation since
#             the model can produce infinite mean times? (See TODO above).
#           -

class ExpPredBroadcaster(OM.Broadcaster):
    """This is a broadcaster which follows the intensity function as defined by
    RMTPP paper and updates the hidden state upon receiving each event by predicting
    the (known) intensity function of the next broadcaster.
    """

    @Deco.optioned()
    def __init__(self, src_id, seed, trainer, t_min=0):
        super(ExpPredBroadcaster, self).__init__(src_id, seed)
        self.init = False

        self.trainer = trainer

        params = Deco.Options(**self.trainer.sess.run({
            'wt': trainer.tf_wt,
            'vt': trainer.tf_vt,
            'bt': trainer.tf_bt,
            'init_h': trainer.tf_h
        }))

        self.cur_h = params.init_h

        self.exp_sampler = ExpCDFSampler(_opts=params,
                                         t_min=t_min,
                                         seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here
        # Actually best done by requesting the trainer to change the state.
        # This couples the broadcaster and the trainer together rather badly,
        # doesn't it? Maybe not so much.
        r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)
        return self.trainer._update_hidden_state(src_id, time_delta, r_t)

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            # Nothing special to do for the first event.

        self.state.apply_event(event)

        if event is None:
            # This is the first event. Post immediately to join the party?
            # Or hold off?
            return self.exp_sampler.generate_sample()
        else:
            self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
            next_post_time = self.exp_sampler.register_event(
                                    event.cur_time,
                                    self.cur_h,
                                    own_event=event.src_id == self.src_id)
            next_delta = next_post_time - self.last_self_event_time
            # print(next_delta)
            assert next_delta >= 0
            return next_delta


OM.SimOpts.registerSource('ExpPredBroadcaster', ExpPredBroadcaster)


class ExpPredTrainer:

    def _update_hidden_state(self, src_id, time_delta, r_t):

        feed_dict = {
            self.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
            self.tf_t_delta: np.asarray([time_delta]).reshape(-1),
            self.tf_h: self.cur_h,
            self.tf_rank: np.asarray([np.mean(r_t)]).reshape(-1)
        }
        return self.sess.run(self.trainer.tf_h_next,
                                     feed_dict=feed_dict)

    @Deco.optioned()
    def __init__(self, Wm, Wl, Wdt, Wt, Wr, Bh, vt, wt, bt, init_h, init_l, init_pred_dt,
                 sess, sim_opts, scope=None, t_min=0, batch_size=16, max_events=100):
        """Initialize the trainer with the policy parameters."""

        self.src_embed_map = {x.src_id: idx + 1
                              for idx, x in enumerate(sim_opts.create_other_sources())}
        self.src_embed_map[sim_opts.src_id] = 0

        self.batch_size = batch_size
        self.max_events = max_events
        self.num_hidden_states = init_h.shape[0]
        self.sim_opts = sim_opts
        self.src_id = sim_opts.src_id
        self.sess = sess

        init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))
        # TODO: May need to do the same for init_l, init_pred_dt

        self.scope = scope or type(self).__name__

        # TODO: Create all these variables on the CPU and the training vars on the GPU
        # by using tf.device explicitly?
        with tf.variable_scope(self.scope):
            with tf.variable_scope("hidden_state"):
                self.tf_Wm  = tf.get_variable(name="Wm", shape=Wm.shape,
                                              initializer=tf.constant_initializer(Wm))
                self.tf_Wl  = tf.get_variable(name="Wl", shape=Wl.shape,
                                              initializer=tf.constant_initializer(Wl))
                self.tf_Wdt = tf.get_variable(name="Wdt", shape=Wdt.shape,
                                              initializer=tf.constant_initializer(Wdt))
                self.tf_Wt  = tf.get_variable(name="Wt", shape=Wt.shape,
                                              initializer=tf.constant_initializer(Wt))
                self.tf_Wr  = tf.get_variable(name="Wr", shape=Wr.shape,
                                              initializer=tf.constant_initializer(Wr))
                self.tf_Bh  = tf.get_variable(name="Bh", shape=Bh.shape,
                                             initializer=tf.constant_initializer(Bh))

                self.tf_l   = tf.get_variable(name="l", shape=init_l.shape,
                                              initializer=tf.constant_initializer(init_l))
                self.tf_pred_dt  = tf.get_variable(name="pred_dt", shape=init_pred_dt.shape,
                                              initializer=tf.constant_initializer(init_pred_dt))
                self.tf_h = tf.get_variable(name="h", shape=(self.num_hidden_states, 1),
                                            initializer=tf.constant_initializer(init_h))
                self.tf_b_idx = tf.placeholder(name="b_idx", shape=1, dtype=tf.int32)
                self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=tf.float32)
                self.tf_rank = tf.placeholder(name="rank", shape=1, dtype=tf.float32)

                # TODO: The transposes hurt my eyes and the GPU efficiency.
                self.tf_h_next = tf.nn.relu(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name="b_embed")
                    ) +
                    tf.matmul(self.tf_Wl, self.tf_l) +
                    tf.matmul(self.tf_Wdt, self.tf_pred_dt) +
                    self.tf_Wr * self.tf_rank +
                    self.tf_Wt * self.tf_t_delta +
                    self.tf_Bh,
                    name="h_next"
                )

            with tf.variable_scope("output"):
                self.tf_bt = tf.get_variable(name="bt", shape=bt.shape,
                                             initializer=tf.constant_initializer(bt))
                self.tf_vt = tf.get_variable(name="vt", shape=vt.shape,
                                             initializer=tf.constant_initializer(vt))
                self.tf_wt = tf.get_variable(name="wt", shape=wt.shape,
                                             initializer=tf.constant_initializer(wt))
                # self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=tf.float32)
                # self.tf_u_t = tf.exp(
                #     tf.tensordot(self.tf_vt, self.tf_h, axes=1) +
                #     self.tf_t_delta * self.tf_wt +
                #     self.tf_bt,
                #     name="u_t"
                # )

            # Create a large dynamic_rnn kind of network which can calculate
            # the gradients for a given given batch of simulations.
            with tf.variable_scope("training"):
                self.tf_batch_rewards = tf.placeholder(name="rewards",
                                                 shape=(batch_size, 1),
                                                 dtype=tf.float32)
                self.tf_batch_t_deltas = tf.placeholder(name="t_deltas",
                                                  shape=(batch_size, max_events),
                                                  dtype=tf.float32)
                self.tf_batch_pred_dt = tf.placeholder(name="pred_dt",
                                                       shape=(batch_size,
                                                              len(self.sim_opts.other_sources),
                                                              max_events),
                                                       dtype=tf.float32)
                self.tf_batch_l       = tf.placeholder(name="ls",
                                                       shape=(batch_size,
                                                              len(self.sim_opts.other_sources),
                                                              max_events),
                                                       dtype=tf.float32)
                self.tf_batch_b_idxes = tf.placeholder(name="b_idxes",
                                                 shape=(batch_size, max_events),
                                                 dtype=tf.int32)
                self.tf_batch_ranks = tf.placeholder(name="ranks",
                                               shape=(batch_size, max_events),
                                               dtype=tf.float32)
                self.tf_batch_seq_len = tf.placeholder(name="seq_len",
                                                 shape=(batch_size, 1),
                                                 dtype=tf.int32)
                self.tf_batch_last_interval = tf.placeholder(name="last_interval",
                                                             shape=batch_size,
                                                             dtype=tf.float32)

                self.tf_batch_init_h = tf_batch_h_t = tf.zeros(name="init_h",
                                              shape=(batch_size, self.num_hidden_states),
                                              dtype=tf.float32)

                self.LL = tf.zeros(name="log_likelihood", dtype=tf.float32, shape=(batch_size))
                self.loss = tf.zeros(name="loss", dtype=tf.float32, shape=(batch_size))

                t_0 = tf.zeros(name="event_time", shape=batch_size, dtype=tf.float32)

                def batch_u_theta(batch_t_deltas):
                    return tf.exp(
                            tf.matmul(tf_batch_h_t, self.tf_vt) +
                            self.tf_wt * tf.expand_dims(batch_t_deltas, 1) +
                            self.tf_bt
                        )


                # TODO: Convert this to a tf.while_loop, perhaps.
                # The performance benefit is debatable.
                for evt_idx in range(max_events):
                    tf_batch_h_t = tf.where(
                        tf.tile(evt_idx <= self.tf_batch_seq_len, [1, self.num_hidden_states]),
                        tf.nn.relu(
                            tf.nn.embedding_lookup(self.tf_Wm,
                                                   self.tf_batch_b_idxes[:, evt_idx]) +
                            tf.matmul(tf_batch_h_t, self.tf_Wh, transpose_b=True) +
                            tf.matmul(tf.expand_dims(self.tf_batch_ranks[:, evt_idx], 1),
                                      self.tf_Wr, transpose_b=True) +
                            tf.matmul(tf.expand_dims(self.tf_batch_t_deltas[:, evt_idx], 1),
                                      self.tf_Wt, transpose_b=True) +
                            tf.tile(tf.transpose(self.tf_Bh), [batch_size, 1])
                        ),
                        tf.zeros(dtype=tf.float32, shape=(batch_size, self.num_hidden_states))
                        # The gradient of a constant w.r.t. a variable is None or 0
                    )
                    tf_batch_u_theta = tf.where(
                        evt_idx <= self.tf_batch_seq_len,
                        batch_u_theta(self.tf_batch_t_deltas[:, evt_idx]),
                        tf.zeros(dtype=tf.float32, shape=(batch_size, 1))
                    )

                    self.LL += tf.where(tf.squeeze(evt_idx <= self.tf_batch_seq_len),
                                    tf.where(tf.equal(self.tf_batch_b_idxes[:, evt_idx], sim_opts.src_id),
                                        tf.squeeze(tf.log(tf_batch_u_theta)),
                                        tf.zeros(dtype=tf.float32, shape=batch_size)) +
                                    (1 / self.tf_wt) * tf.squeeze(
                                        batch_u_theta(t_0) -
                                        tf_batch_u_theta
                                    ),
                                    tf.zeros(dtype=tf.float32, shape=batch_size))

                    self.loss += tf.where(tf.squeeze(evt_idx <= self.tf_batch_seq_len),
                                    -(1 / (2 * self.tf_wt)) * tf.squeeze(
                                        tf.square(batch_u_theta(t_0)) -
                                        tf.square(tf_batch_u_theta)
                                    ),
                                    tf.zeros(dtype=tf.float32, shape=(batch_size)))

        # Here, outside the loop, add the survival term for the batch to
        # both the loss and to the LL.
        self.LL += (1 / self.tf_wt) * tf.squeeze(
            batch_u_theta(t_0) - batch_u_theta(self.tf_batch_last_interval)
        )
        self.loss += - (1 / (2 * self.tf_wt)) * tf.squeeze(
            tf.square(batch_u_theta(t_0)) - tf.square(self.tf_batch_last_interval)
        )

        # sim_feed_dict = {
        #     self.tf_Wm: Wm,
        #     self.tf_Wh: Wh,
        #     self.tf_Wt: Wt,
        #     self.tf_Bh: Bh,

        #     self.tf_bt: bt,
        #     self.tf_vt: vt,
        #     self.tf_wt: wt,
        # }

    def initialize(self):
        """Initialize the graph."""
        self.sess.run(tf.global_variables_initializer())
        # No more nodes will be added to the graph beyond this point.
        # Recommended way to prevent memory leaks afterwards, esp. if the
        # session will be used in a multi-threaded manner.
        # https://stackoverflow.com/questions/38694111/
        self.sess.graph.finalize()

    def _create_exp_broadcaster(self, seed):
        """Create a new exp_broadcaster with the current params."""
        return ExpPredBroadcaster(src_id=self.src_id, seed=seed, trainer=self)

    def run_sim(self, seed):
        """Run one simulation and return the dataframe.
        Will be thread-safe and can be called multiple times."""
        run_sim_opts = self.sim_opts.update({})
        exp_b = self._create_exp_broadcaster(seed=seed * 3)

        mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)
        mgr.run_dynamic()
        return mgr.get_state().get_dataframe()

    def reward_fn(self, df):
        """Calculate the reward for a given trajectory."""
        rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
        rank_dt = np.diff(np.concatenate([rank_in_tau.index.values,
                                          [self.sim_opts.end_time]]))
        return np.sum((rank_in_tau ** 2) * rank_dt)

    def get_feed_dict(self, batch_df):
        """Produce a feed_dict for the given batch."""
        assert all(len(df.sink_id.unique()) == 1 for df in batch_df), "Can only handle one sink at the moment."
        assert len(batch_df) == self.batch_size, "The batch should consist of {} simulations, not {}.".format(self.batch_size, len(batch_df))

        full_shape = (self.batch_size, self.max_events)

        batch_rewards = np.asarray([self.reward_fn(x) for x in batch_df])[:, np.newaxis]
        batch_t_deltas = np.zeros(shape=full_shape, dtype=float)

        batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
        batch_ranks = np.zeros(shape=full_shape, dtype=float)
        batch_seq_len = np.asarray([np.minimum(x.shape[0], self.max_events) for x in batch_df], dtype=int)[:, np.newaxis]
        batch_init_h = np.zeros(shape=(self.batch_size, self.num_hidden_states), dtype=float)

        batch_last_interval = np.zeros(shape=self.batch_size, dtype=float)

        for idx, df in enumerate(batch_df):
            # They are sorted by time already.
            batch_len = int(batch_seq_len[idx])
            rank_in_tau = RU.rank_of_src_in_df(df=df, src_id=self.src_id).mean(axis=1)
            batch_ranks[idx, 0:batch_len] = rank_in_tau.values[0:batch_len]
            batch_b_idxes[idx, 0:batch_len] = df.src_id.map(self.src_embed_map).values[0:batch_len]
            batch_t_deltas[idx, 0:batch_len] = df.time_delta.values[0:batch_len]
            if batch_len == df.shape[0]:
                # This batch has consumed all the events
                batch_last_interval[idx] = self.sim_opts.end_time - df.t.iloc[-1]
            else:
                batch_last_interval[idx] = df.time_delta[batch_len]

        return {
            self.tf_batch_b_idxes: batch_b_idxes,
            self.tf_batch_rewards: batch_rewards,
            self.tf_batch_seq_len: batch_seq_len,
            self.tf_batch_t_deltas: batch_t_deltas,
            self.tf_batch_ranks: batch_ranks,
            self.tf_batch_init_h: batch_init_h,
            self.tf_batch_last_interval: batch_last_interval,
        }

    def calc_grad(self, df):
        """Calculate the gradient with respect to a certain run."""
        # 1. Keep updating the u_{\theta}(t) in the tensorflow graph starting from
        #    t = 0 with each event and calculating the gradient.
        # 2. Finally, sum together the gradient calculated for the complete
        #    sequence.

        # Actually, we can calculate the gradient analytically in this case.
        # Not quite: we can integrate analytically, but differentiation is
        # still a little tricky because of the hidden state.
        R_tau = self.reward_fn(df, src_id=self.src_id)

        # Loop over the events.
        unique_events = df.groupby('event_id').first()
        for t_delta, src_id in unique_events[['time_delta', 'src_id']].values:
            # TODO
            pass
