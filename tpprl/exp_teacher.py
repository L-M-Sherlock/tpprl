import numpy as np
import os
import tensorflow as tf
import decorated_options as Deco
import warnings

SAVE_DIR = 'teacher-log'
MAX_EVENTS = 100000

# DEBUG ONLY
try:
    from .utils import variable_summaries, _now
    from .cells import TPPRExpMarkedCellStacked
    from .exp_sampler import ExpCDFSampler
except ModuleNotFoundError:
    warnings.warn('Could not import local modules. Assuming they have been loaded using %run -i')


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


class Student:
    def __init__(self, n_0s, alphas, betas, seed):
        """n_0 is the initial expertise in all items."""
        self.ns = n_0s
        self.alphas = alphas
        self.betas = betas
        self.seed = seed
        self.last_review_times = np.zeros_like(n_0s)
        self.RS = np.random.RandomState(seed)

    def review(self, item, cur_time):
        recall = self.recall(item, cur_time)
        if recall:
            self.ns[item] *= (1 - self.alphas[item])
        else:
            self.ns[item] *= (1 + self.betas[item])

        # The numbers need to be more carefully tuned.
        self.ns[item] = min(max(1e-6, self.ns[item]), 1e2)

        self.last_review_times[item] = cur_time
        return recall

    def recall(self, item, t):
        m_t = np.exp(-(self.ns[item] * (t - self.last_review_times[item])))
        return self.RS.rand() < m_t


class Scenario:
    def __init__(self, alphas, betas, seed,
                 Wm, Wh, Wr, Wt, Bh, Vy,
                 wt, vt, bt, init_h, T):

        n_0s = np.ones_like(alphas)
        self.num_items = n_0s.shape[0]

        self.student = Student(n_0s=n_0s, alphas=alphas, betas=betas, seed=seed * 2)
        self.Wm = Wm
        self.Wh = Wh
        self.Wr = Wr
        self.Wt = Wt
        self.Bh = Bh
        self.Vy = Vy

        self._init = False
        self.cur_h = init_h
        self.T = T
        self.RS = np.random.RandomState(seed)
        self.last_time = -1

        self.c_is = []
        self.hidden_states = []
        self.time_deltas = []
        self.recalls = []
        self.items = []

        self.params = Deco.Options(**{
            'wt': wt,
            'vt': vt,
            'bt': bt,
            'init_h': init_h
        })

        self.exp_sampler = ExpCDFSampler(_opts=self.params,
                                         t_min=0,
                                         seed=seed + 1)

    def get_all_c_is(self):
        assert self._init
        return np.asarray(self.c_is + [self.exp_sampler.c])

    def get_last_interval(self):
        assert self._init
        return self.T - self.last_time

    def get_all_time_deltas(self):
        assert self._init
        return np.asarray(self.time_deltas +
                          [self.T - self.last_time])

    def get_all_hidden_states(self):
        assert self._init
        return np.asarray(self.hidden_states + [self.cur_h])

    def get_num_events(self):
        assert self._init
        return len(self.c_is)

    def update_hidden_state(self, item, t, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        recall = float(self.student.review(item, t))
        self.recalls.append(recall)

        return np.tanh(
            self.Wm[item, :][:, np.newaxis] +
            self.Wh.dot(self.cur_h) +
            self.Wr.dot(recall) +
            self.Wt * time_delta +
            self.Bh
        )

    def generate_sample(self):
        t_next = self.exp_sampler.generate_sample()
        p = softmax(self.Vy.T.dot(self.cur_h)).squeeze(axis=-1)
        item_next = self.RS.choice(np.arange(self.num_items), p=p)
        return (t_next, item_next)

    def run(self, max_events=None):
        """Execute a study episode."""
        assert not self._init
        self._init = True

        if max_events is None:
            max_events = float('inf')

        idx = 0
        t = 0
        (t_next, item_next) = self.generate_sample()

        while idx < max_events and t_next < self.T:
            idx += 1
            time_delta = t_next - t

            self.items.append(item_next)
            self.c_is.append(self.exp_sampler.c)
            self.hidden_states.append(self.cur_h)
            self.time_deltas.append(time_delta)
            self.last_time = t

            t = t_next

            self.cur_h = self.update_hidden_state(item_next, t, time_delta)
            self.exp_sampler.register_event(t, self.cur_h, own_event=True)
            (t_next, item_next) = self.generate_sample()

    def reward(self, tau):
        """Returns the result of a test conducted at T + tau."""
        return np.mean([self.student.recall(item, self.T + tau)
                       for item in range(self.num_items)])


def mk_def_teacher_opts(hidden_dims, num_items,
                        scenario_opts, seed=42, **kwargs):
    """Make default option set."""
    RS  = np.random.RandomState(seed=seed)

    def_exp_recurrent_teacher_opts = Deco.Options(
        t_min=0,
        scope=None,
        decay_steps=100,
        decay_rate=0.001,
        num_hidden_states=hidden_dims,
        learning_rate=.01,
        clip_norm=1.0,
        tau=15.0,

        Wh=RS.randn(hidden_dims, hidden_dims) * 0.1 + np.diag(np.ones(hidden_dims)),  # Careful initialization
        Wm=RS.randn(num_items, hidden_dims),
        Wr=RS.randn(hidden_dims, 1),
        Wt=RS.randn(hidden_dims, 1),
        Vy=RS.randn(hidden_dims, num_items),
        Bh=RS.randn(hidden_dims, 1),

        vt=RS.randn(hidden_dims, 1),
        wt=np.abs(RS.rand(1)) * -1,
        bt=np.abs(RS.randn(1)),

        # The graph execution time depends on this parameter even though each
        # trajectory may contain much fewer events. So it is wise to set
        # it such that it is just above the total number of events likely
        # to be seen.
        momentum=0.9,
        max_events=5000,
        batch_size=16,
        end_time=100.0,

        device_cpu='/cpu:0',
        device_gpu='/gpu:0',
        only_cpu=False,

        save_dir=SAVE_DIR,

        # Expected: './tpprl.summary/train-{}/'.format(run)
        summary_dir=None,

        decay_q_rate=0.0,

        # Whether or not to use the advantage formulation.
        with_advantage=True,

        q=0.001,

        scenario_opts=scenario_opts,
    )

    return def_exp_recurrent_teacher_opts.set(**kwargs)


class ExpRecurrentTeacher:
    @Deco.optioned()
    def __init__(self, Vy, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, num_hidden_states,
                 sess, scope, batch_size, max_events, q,
                 learning_rate, clip_norm, t_min, end_time,
                 summary_dir, save_dir, decay_steps, decay_rate, momentum,
                 device_cpu, device_gpu, only_cpu, with_advantage,
                 num_items, decay_q_rate, scenario_opts, tau):
        """Initialize the trainer with the policy parameters."""

        self.decay_q_rate = decay_q_rate
        self.scenario_opts = scenario_opts

        self.t_min = 0
        self.t_max = end_time
        self.tau = tau

        self.summary_dir = summary_dir
        self.save_dir = save_dir

        # self.src_embed_map = {x.src_id: idx + 1
        #                       for idx, x in enumerate(sim_opts.create_other_sources())}

        # To handle multiple reloads of redqueen related modules.
        self.src_embed_map = np.arange(num_items)

        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.clip_norm = clip_norm

        self.q = q
        self.batch_size = batch_size

        self.tf_batch_size = None
        self.tf_max_events = None
        self.num_items = num_items

        self.abs_max_events = max_events
        self.num_hidden_states = num_hidden_states

        # init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))

        self.scope = scope or type(self).__name__

        var_device = device_cpu if only_cpu else device_gpu

        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        with tf.device(device_cpu):
            # Global step needs to be on the CPU (Why?)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('hidden_state'):
                with tf.device(var_device):
                    self.tf_Wm = tf.get_variable(name='Wm', shape=Wm.shape,
                                                 initializer=tf.constant_initializer(Wm))
                    self.tf_Wh = tf.get_variable(name='Wh', shape=Wh.shape,
                                                 initializer=tf.constant_initializer(Wh))
                    self.tf_Wt = tf.get_variable(name='Wt', shape=Wt.shape,
                                                 initializer=tf.constant_initializer(Wt))
                    self.tf_Wr = tf.get_variable(name='Wr', shape=Wr.shape,
                                                 initializer=tf.constant_initializer(Wr))
                    self.tf_Bh = tf.get_variable(name='Bh', shape=Bh.shape,
                                                 initializer=tf.constant_initializer(Bh))

                    # Needed to calculate the hidden state for one step.
                    self.tf_h = tf.get_variable(name='h', initializer=tf.zeros((self.num_hidden_states, 1), dtype=self.tf_dtype))

                self.tf_b_idx = tf.placeholder(name='b_idx', shape=1, dtype=tf.int32)
                self.tf_t_delta = tf.placeholder(name='t_delta', shape=1, dtype=self.tf_dtype)
                self.tf_recall = tf.placeholder(name='recall', shape=(1, 1), dtype=self.tf_dtype)

                self.tf_h_next = tf.nn.tanh(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name='b_embed')
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) +
                    tf.matmul(self.tf_Wr, self.tf_recall) +
                    self.tf_Wt * self.tf_t_delta +
                    self.tf_Bh,
                    name='h_next'
                )

            with tf.variable_scope('output'):
                with tf.device(var_device):
                    self.tf_Vy = tf.get_variable(name='Vy', shape=Vy.shape,
                                                 initializer=tf.constant_initializer(Vy))
                    self.tf_bt = tf.get_variable(name='bt', shape=bt.shape,
                                                 initializer=tf.constant_initializer(bt))
                    self.tf_vt = tf.get_variable(name='vt', shape=vt.shape,
                                                 initializer=tf.constant_initializer(vt))
                    self.tf_wt = tf.get_variable(name='wt', shape=wt.shape,
                                                 initializer=tf.constant_initializer(wt))
                # self.tf_t_delta = tf.placeholder(name='t_delta', shape=1, dtype=self.tf_dtype)
                # self.tf_u_t = tf.exp(
                #     tf.tensordot(self.tf_vt, self.tf_h, axes=1) +
                #     self.tf_t_delta * self.tf_wt +
                #     self.tf_bt,
                #     name='u_t'
                # )

            # Create a large dynamic_rnn kind of network which can calculate
            # the gradients for a given given batch of simulations.
            with tf.variable_scope('training'):
                self.tf_batch_rewards = tf.placeholder(name='rewards',
                                                       shape=(self.tf_batch_size, 1),
                                                       dtype=self.tf_dtype)
                self.tf_batch_t_deltas = tf.placeholder(name='t_deltas',
                                                        shape=(self.tf_batch_size, self.tf_max_events),
                                                        dtype=self.tf_dtype)
                self.tf_batch_b_idxes = tf.placeholder(name='b_idxes',
                                                       shape=(self.tf_batch_size, self.tf_max_events),
                                                       dtype=tf.int32)
                self.tf_batch_recalls = tf.placeholder(name='recalls',
                                                       shape=(self.tf_batch_size, self.tf_max_events),
                                                       dtype=self.tf_dtype)
                self.tf_batch_seq_len = tf.placeholder(name='seq_len',
                                                       shape=(self.tf_batch_size, 1),
                                                       dtype=tf.int32)
                self.tf_batch_last_interval = tf.placeholder(name='last_interval',
                                                             shape=self.tf_batch_size,
                                                             dtype=self.tf_dtype)

                # Inferred batch size
                inf_batch_size = tf.shape(self.tf_batch_b_idxes)[0]

                self.tf_batch_init_h = tf.zeros(
                    name='init_h',
                    shape=(inf_batch_size, self.num_hidden_states),
                    dtype=self.tf_dtype
                )

                # Stacked version (for performance)

                with tf.name_scope('stacked'):
                    with tf.device(var_device):
                        (self.Wm_mini, self.Wr_mini, self.Wh_mini,
                         self.Wt_mini, self.Bh_mini, self.wt_mini,
                         self.vt_mini, self.bt_mini, self.Vy_mini) = [
                             tf.stack(x, name=name)
                             for x, name in zip(
                                     zip(*[
                                         (tf.identity(self.tf_Wm), tf.identity(self.tf_Wr),
                                          tf.identity(self.tf_Wh), tf.identity(self.tf_Wt),
                                          tf.identity(self.tf_Bh), tf.identity(self.tf_wt),
                                          tf.identity(self.tf_vt), tf.identity(self.tf_bt),
                                          tf.identity(self.tf_Vy))
                                         for _ in range(self.batch_size)
                                     ]),
                                     ['Wm', 'Wr', 'Wh', 'Wt', 'Bh', 'wt', 'vt', 'bt', 'Vy']
                             )
                        ]

                        self.rnn_cell_stack = TPPRExpMarkedCellStacked(
                            hidden_state_size=(None, self.num_hidden_states),
                            output_size=[self.num_hidden_states] + [1] * 3,
                            tf_dtype=self.tf_dtype,
                            Wm=self.Wm_mini, Wr=self.Wr_mini,
                            Wh=self.Wh_mini, Wt=self.Wt_mini,
                            Bh=self.Bh_mini, wt=self.wt_mini,
                            vt=self.vt_mini, bt=self.bt_mini,
                            Vy=self.Vy_mini
                        )

                        ((self.h_states_stack, LL_log_terms_stack, LL_int_terms_stack, loss_terms_stack), tf_batch_h_t_mini) = tf.nn.dynamic_rnn(
                            self.rnn_cell_stack,
                            inputs=(tf.expand_dims(self.tf_batch_b_idxes, axis=-1),
                                    tf.expand_dims(self.tf_batch_recalls, axis=-1),
                                    tf.expand_dims(self.tf_batch_t_deltas, axis=-1)),
                            sequence_length=tf.squeeze(self.tf_batch_seq_len, axis=-1),
                            dtype=self.tf_dtype,
                            initial_state=self.tf_batch_init_h
                        )

                        self.LL_log_terms_stack = tf.squeeze(LL_log_terms_stack, axis=-1)
                        self.LL_int_terms_stack = tf.squeeze(LL_int_terms_stack, axis=-1)
                        self.loss_terms_stack = tf.squeeze(loss_terms_stack, axis=-1)

                        # LL_last_term_stack = rnn_cell.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                        # loss_last_term_stack = rnn_cell.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                        self.LL_last_term_stack = self.rnn_cell_stack.last_LL(tf_batch_h_t_mini, self.tf_batch_last_interval)
                        self.loss_last_term_stack = self.rnn_cell_stack.last_loss(tf_batch_h_t_mini, self.tf_batch_last_interval)

                        self.LL_stack = (tf.reduce_sum(self.LL_log_terms_stack, axis=1) - tf.reduce_sum(self.LL_int_terms_stack, axis=1)) + self.LL_last_term_stack
                        self.loss_stack = (self.q / 2) * (tf.reduce_sum(self.loss_terms_stack, axis=1) + self.loss_last_term_stack) * tf.pow(tf.cast(self.global_step, self.tf_dtype), self.decay_q_rate)

            with tf.name_scope('calc_u'):
                with tf.device(var_device):
                    # These are operations needed to calculate u(t) in post-processing.
                    # These can be done entirely in numpy-space, but since we have a
                    # version in tensorflow, they have been moved here to avoid
                    # memory leaks.
                    # Otherwise, new additions to the graph were made whenever the
                    # function calc_u was called.

                    self.calc_u_h_states = tf.placeholder(
                        name='calc_u_h_states',
                        shape=(self.tf_batch_size, self.tf_max_events, self.num_hidden_states),
                        dtype=self.tf_dtype
                    )
                    self.calc_u_batch_size = tf.placeholder(
                        name='calc_u_batch_size',
                        shape=(None,),
                        dtype=tf.int32
                    )

                    self.calc_u_c_is_init = tf.matmul(self.tf_batch_init_h, self.tf_vt) + self.tf_bt
                    self.calc_u_c_is_rest = tf.squeeze(
                        tf.matmul(
                            self.calc_u_h_states,
                            tf.tile(
                                tf.expand_dims(self.tf_vt, 0),
                                [self.calc_u_batch_size[0], 1, 1]
                            )
                        ) + self.tf_bt,
                        axis=-1,
                        name='calc_u_c_is_rest'
                    )

                    self.calc_u_is_own_event = tf.equal(self.tf_batch_b_idxes, 0)

        # TODO: The all_tf_vars and all_mini_vars MUST be kept in sync.
        self.all_tf_vars = [self.tf_Wh, self.tf_Wm, self.tf_Wt, self.tf_Bh,
                            self.tf_Wr, self.tf_bt, self.tf_vt, self.tf_wt, self.tf_Vy]

        self.all_mini_vars = [self.Wh_mini, self.Wm_mini, self.Wt_mini, self.Bh_mini,
                              self.Wr_mini, self.bt_mini, self.vt_mini, self.wt_mini, self.Vy_mini]

        with tf.name_scope('stack_grad'):
            with tf.device(var_device):
                self.LL_grad_stacked = {x: tf.gradients(self.LL_stack, x)
                                        for x in self.all_mini_vars}
                self.loss_grad_stacked = {x: tf.gradients(self.loss_stack, x)
                                          for x in self.all_mini_vars}

                self.avg_gradient_stack = []

                # TODO: Can we calculate natural gradients here easily?
                # TODO: Should we take into account the loss as well as the reward?
                # This is one of the baseline rewards we can calculate.
                avg_reward = tf.reduce_mean(self.tf_batch_rewards, axis=0) + tf.reduce_mean(self.loss_stack, axis=0) if with_advantage else 0.0

                # Removing the average reward converts this coefficient into the advantage function.
                coef = tf.squeeze(self.tf_batch_rewards, axis=-1) + self.loss_stack - avg_reward

                for x, y in zip(self.all_mini_vars, self.all_tf_vars):
                    LL_grad = self.LL_grad_stacked[x][0]
                    if x == self.Vy_mini:
                        loss_grad = 0
                    else:
                        loss_grad = self.loss_grad_stacked[x][0]

                    dim = len(LL_grad.get_shape())
                    if dim == 1:
                        self.avg_gradient_stack.append(
                            (tf.reduce_mean(LL_grad * coef + loss_grad, axis=0), y)
                        )
                    elif dim == 2:
                        self.avg_gradient_stack.append(
                            (
                                tf.reduce_mean(
                                    LL_grad * tf.tile(tf.reshape(coef, (-1, 1)),
                                                      [1, tf.shape(LL_grad)[1]]) +
                                    loss_grad,
                                    axis=0
                                ),
                                y
                            )
                        )
                    elif dim == 3:
                        self.avg_gradient_stack.append(
                            (
                                tf.reduce_mean(
                                    LL_grad * tf.tile(tf.reshape(coef, (-1, 1, 1)),
                                                      [1, tf.shape(LL_grad)[1], tf.shape(LL_grad)[2]]) +
                                    loss_grad,
                                    axis=0
                                ),
                                y
                            )
                        )

                self.clipped_avg_gradients_stack, self.grad_norm_stack = \
                    tf.clip_by_global_norm([grad for grad, _ in self.avg_gradient_stack],
                                           clip_norm=self.clip_norm)

                self.clipped_avg_gradient_stack = list(zip(
                    self.clipped_avg_gradients_stack,
                    [var for _, var in self.avg_gradient_stack]
                ))

        self.tf_learning_rate = tf.train.inverse_time_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate
        )

        self.opt = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate,
                                          beta1=momentum)
        self.sgd_stacked_op = self.opt.apply_gradients(self.clipped_avg_gradient_stack,
                                                       global_step=self.global_step)

        self.sess = sess

        # There are other global variables as well, like the ones which the
        # ADAM optimizer uses.
        self.saver = tf.train.Saver(tf.global_variables(),
                                    keep_checkpoint_every_n_hours=0.25,
                                    max_to_keep=1000)

        with tf.device(device_cpu):
            tf.contrib.training.add_gradients_summaries(self.avg_gradient_stack)

            for v in self.all_tf_vars:
                variable_summaries(v)

            variable_summaries(self.tf_learning_rate, name='learning_rate')
            variable_summaries(self.loss_stack, name='loss_stack')
            variable_summaries(self.LL_stack, name='LL_stack')
            variable_summaries(self.loss_last_term_stack, name='loss_last_term_stack')
            variable_summaries(self.LL_last_term_stack, name='LL_last_term_stack')
            variable_summaries(self.h_states_stack, name='hidden_states_stack')
            variable_summaries(self.LL_log_terms_stack, name='LL_log_terms_stack')
            variable_summaries(self.LL_int_terms_stack, name='LL_int_terms_stack')
            variable_summaries(self.loss_terms_stack, name='loss_terms_stack')
            variable_summaries(tf.cast(self.tf_batch_seq_len, self.tf_dtype),
                               name='batch_seq_len')

            self.tf_merged_summaries = tf.summary.merge_all()

    def initialize(self, finalize=True):
        """Initialize the graph."""
        self.sess.run(tf.global_variables_initializer())
        if finalize:
            # No more nodes will be added to the graph beyond this point.
            # Recommended way to prevent memory leaks afterwards, esp. if the
            # session will be used in a multi-threaded manner.
            # https://stackoverflow.com/questions/38694111/
            self.sess.graph.finalize()

    def train_many(self, num_iters, init_seed=42, with_summaries=False):
        """Run one SGD op given a batch of simulation."""

        seed_start = init_seed + self.sess.run(self.global_step) * self.batch_size

        if with_summaries:
            assert self.summary_dir is not None
            os.makedirs(self.summary_dir, exist_ok=True)
            train_writer = tf.summary.FileWriter(self.summary_dir,
                                                 self.sess.graph)
        train_op = self.sgd_stacked_op
        grad_norm_op = self.grad_norm_stack
        LL_op = self.LL_stack
        loss_op = self.loss_stack

        for iter_idx in range(num_iters):
            seed_end = seed_start + self.batch_size

            seeds = range(seed_start, seed_end)
            scenarios = [run_scenario(self, seed) for seed in seeds]

            num_events = [s.get_num_events() for s in scenarios]
            f_d = get_feed_dict(self, scenarios)

            if with_summaries:
                reward, LL, loss, grad_norm, summaries, step, lr, _ = \
                    self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                   grad_norm_op, self.tf_merged_summaries,
                                   self.global_step, self.tf_learning_rate,
                                   train_op],
                                  feed_dict=f_d)
                train_writer.add_summary(summaries, step)
            else:
                reward, LL, loss, grad_norm, step, lr, _ = \
                    self.sess.run([self.tf_batch_rewards, LL_op, loss_op,
                                   grad_norm_op, self.global_step,
                                   self.tf_learning_rate, train_op],
                                  feed_dict=f_d)

            mean_LL = np.mean(LL)
            mean_loss = np.mean(loss)
            mean_reward = np.mean(reward)

            print('{} Run {}, LL {:.5f}, loss {:.5f}, Rwd {:.5f}'
                  ', CTG {:.5f}, seeds {}--{}, grad_norm {:.5f}, step = {}'
                  ', lr = {:.5f}, events = {:.2f}'
                  .format(_now(), iter_idx, mean_LL, mean_loss,
                          mean_reward, mean_reward + mean_loss,
                          seed_start, seed_end - 1, grad_norm, step, lr,
                          np.mean(num_events)))

            # Ready for the next iter_idx.
            seed_start = seed_end

        chkpt_file = os.path.join(self.save_dir, 'tpprl.ckpt')
        self.saver.save(self.sess, chkpt_file, global_step=self.global_step,)

    def restore(self, restore_dir=None, epoch_to_recover=None):
        """Restores the model from a saved checkpoint."""

        if restore_dir is None:
            restore_dir = self.save_dir

        chkpt = tf.train.get_checkpoint_state(restore_dir)

        if epoch_to_recover is not None:
            suffix = '-{}'.format(epoch_to_recover)
            file = [x for x in chkpt.all_model_checkpoint_paths
                    if x.endswith(suffix)]
            if len(file) < 1:
                raise FileNotFoundError('Epoch {} not found.'
                                        .format(epoch_to_recover))
            self.saver.restore(self.sess, file[0])
        else:
            self.saver.restore(self.sess, chkpt.model_checkpoint_path)

    def calc_u(self, h_states, feed_dict, batch_size, times, batch_time_start=None):
        """Calculate u(t) at the times provided."""
        # TODO: May not work if abs_max_events is hit.

        if batch_time_start is None:
            batch_time_start = np.zeros(batch_size)

        feed_dict[self.calc_u_h_states] = h_states
        feed_dict[self.calc_u_batch_size] = [batch_size]

        tf_seq_len = np.squeeze(
            self.sess.run(self.tf_batch_seq_len, feed_dict=feed_dict),
            axis=-1
        ) + 1  # +1 to include the survival term.

        assert self.tf_max_events is None or np.all(tf_seq_len < self.abs_max_events), "Cannot handle events > max_events right now."
        # This will involve changing how the survival term is added, is_own_event is added, etc.

        tf_c_is_arr = self.sess.run(self.calc_u_c_is_rest, feed_dict=feed_dict)
        tf_c_is = (
            [
                self.sess.run(
                    self.calc_u_c_is_init,
                    feed_dict=feed_dict
                )
            ] +
            np.split(tf_c_is_arr, tf_c_is_arr.shape[1], axis=1)
        )
        tf_c_is = list(zip(*tf_c_is))

        tf_t_deltas_arr = self.sess.run(self.tf_batch_t_deltas, feed_dict=feed_dict)
        tf_t_deltas = (
            np.split(tf_t_deltas_arr, tf_t_deltas_arr.shape[1], axis=1) +
            # Cannot add last_interval at the end of the array because
            # the sequence may have ended before that.
            # Instead, we add tf_t_deltas of 0 to make the length of this
            # array the same as of tf_c_is
            [np.asarray([0.0] * batch_size)]
        )
        tf_t_deltas = list(zip(*tf_t_deltas))

        tf_is_own_event_arr = self.sess.run(self.calc_u_is_own_event, feed_dict=feed_dict)
        tf_is_own_event = (
            np.split(tf_is_own_event_arr, tf_is_own_event_arr.shape[1], axis=1) +
            [np.asarray([False] * batch_size)]
        )

        tf_is_own_event = [
            [bool(x) for x in y]
            for y in list(zip(*tf_is_own_event))
        ]

        last_intervals = self.sess.run(
            self.tf_batch_last_interval,
            feed_dict=feed_dict
        )

        for idx in range(batch_size):
            # assert tf_is_own_event[idx][tf_seq_len[idx] - 1]
            tf_is_own_event[idx][tf_seq_len[idx] - 1] = False

            assert tf_t_deltas[idx][tf_seq_len[idx] - 1] == 0

            # This quantity may be zero for real-data.
            # assert tf_t_deltas[idx][tf_seq_len[idx] - 2] > 0

            # tf_t_deltas[idx] is a tuple,
            # we to change it to a list to update a value and then convert
            # back to a tuple.
            old_t_deltas = list(tf_t_deltas[idx])
            old_t_deltas[tf_seq_len[idx] - 1] = last_intervals[idx]
            tf_t_deltas[idx] = tuple(old_t_deltas)

        vt = self.sess.run(self.tf_vt)
        wt = self.sess.run(self.tf_wt)
        bt = self.sess.run(self.tf_bt)

        # TODO: This will break as soon as we move away from zeros
        # as the initial state.
        init_h = np.asarray([0] * self.num_hidden_states)

        sampler_LL = []
        sampler_loss = []

        for idx in range(batch_size):
            # TODO: Split based on the kind of intensity function.

            # The seed doesn't make a difference because we will not
            # take samples from this sampler, we will only ask it to
            # calculate the square loss and the LL.
            #
            # TODO: This sampler needs to change from ExpCDFSampler to
            # SigmoidCDFSampler.
            sampler = ExpCDFSampler(vt=vt, wt=wt, bt=bt,
                                    init_h=init_h,
                                    t_min=batch_time_start[idx],
                                    seed=42)
            sampler_LL.append(
                float(
                    sampler.calc_LL(
                        tf_t_deltas[idx][:tf_seq_len[idx]],
                        tf_c_is[idx][:tf_seq_len[idx]],
                        tf_is_own_event[idx][:tf_seq_len[idx]]
                    )
                )
            )
            sampler_loss.append(
                (self.q / 2) *
                float(
                    sampler.calc_quad_loss(
                        tf_t_deltas[idx][:tf_seq_len[idx]],
                        tf_c_is[idx][:tf_seq_len[idx]]
                    )
                )
            )

        u = np.zeros((batch_size, times.shape[0]), dtype=float)

        for batch_idx in range(batch_size):
            abs_time = batch_time_start[idx]
            abs_idx = 0
            c = tf_c_is[batch_idx][0]

            for time_idx, t in enumerate(times):
                # We do not wish to update the c for the last survival interval.
                # Hence, the -1 in len(tf_t_deltas[batch_idx] - 1
                # while abs_idx < len(tf_t_deltas[batch_idx]) - 1 and abs_time + tf_t_deltas[batch_idx][abs_idx] < t:
                while abs_idx < tf_seq_len[batch_idx] - 1 and abs_time + tf_t_deltas[batch_idx][abs_idx] < t:
                    abs_time += tf_t_deltas[batch_idx][abs_idx]
                    abs_idx += 1
                    c = tf_c_is[batch_idx][abs_idx]

                # TODO: Split based on the kind of intensity function.
                u[batch_idx, time_idx] = np.exp(c + wt * (t - abs_time))

        return {
            'c_is': tf_c_is,
            'is_own_event': tf_is_own_event,
            't_deltas': tf_t_deltas,
            'seq_len': tf_seq_len,

            'vt': vt,
            'wt': wt,
            'bt': bt,

            'LL': sampler_LL,
            'loss': sampler_loss,

            'times': times,
            'u': u,
        }


def get_feed_dict(teacher, scenarios):
    """Produce a feed_dict for the given list of scenarios."""

    # assert all(df.sink_id.nunique() == 1 for df in batch_df), "Can only handle one sink at the moment."

    batch_size = len(scenarios)
    max_events = max(s.get_num_events() for s in scenarios)

    full_shape = (batch_size, max_events)

    batch_rewards = np.asarray([
        -s.reward(teacher.tau) for s in scenarios
    ])[:, np.newaxis]

    batch_last_interval = np.asarray([
        s.get_last_interval() for s in scenarios
    ], dtype=float)

    batch_seq_len = np.asarray([
        s.get_num_events() for s in scenarios
    ], dtype=float)[:, np.newaxis]

    batch_t_deltas = np.zeros(shape=full_shape, dtype=float)
    batch_b_idxes = np.zeros(shape=full_shape, dtype=int)
    batch_recalls = np.zeros(shape=full_shape, dtype=float)
    batch_init_h = np.zeros(shape=(batch_size, teacher.num_hidden_states), dtype=float)

    for idx, scen in enumerate(scenarios):
        # They are sorted by time already.
        batch_len = int(batch_seq_len[idx])

        batch_recalls[idx, 0:batch_len] = scen.recalls
        batch_b_idxes[idx, 0:batch_len] = scen.items
        batch_t_deltas[idx, 0:batch_len] = scen.time_deltas

    return {
        teacher.tf_batch_b_idxes: batch_b_idxes,
        teacher.tf_batch_rewards: batch_rewards,
        teacher.tf_batch_seq_len: batch_seq_len,
        teacher.tf_batch_t_deltas: batch_t_deltas,
        teacher.tf_batch_recalls: batch_recalls,
        teacher.tf_batch_init_h: batch_init_h,
        teacher.tf_batch_last_interval: batch_last_interval,
    }


def mk_scenario_from_opts(teacher_opts, seed):
    alphas = teacher_opts.scenario_opts['alphas']
    betas = teacher_opts.scenario_opts['betas']

    return Scenario(alphas, betas, seed,
                    Wh=teacher_opts.Wh,
                    Wm=teacher_opts.Wm,
                    Wr=teacher_opts.Wr,
                    Wt=teacher_opts.Wt,
                    Vy=teacher_opts.Vy,
                    Bh=teacher_opts.Bh,

                    vt=teacher_opts.vt,
                    wt=teacher_opts.wt,
                    bt=teacher_opts.bt,

                    init_h=np.zeros((teacher_opts.num_hidden_states, 1)),
                    T=100.0)


def mk_scenario_from_teacher(teacher, seed):
    alphas = teacher.scenario_opts['alphas']
    betas = teacher.scenario_opts['betas']

    return Scenario(alphas, betas, seed,
                    Wh=teacher.sess.run(teacher.tf_Wh),
                    Wm=teacher.sess.run(teacher.tf_Wm),
                    Wr=teacher.sess.run(teacher.tf_Wr),
                    Wt=teacher.sess.run(teacher.tf_Wt),
                    Vy=teacher.sess.run(teacher.tf_Vy),
                    Bh=teacher.sess.run(teacher.tf_Bh),

                    vt=teacher.sess.run(teacher.tf_vt),
                    wt=teacher.sess.run(teacher.tf_wt),
                    bt=teacher.sess.run(teacher.tf_bt),

                    init_h=np.zeros((teacher.num_hidden_states, 1)),
                    T=teacher.t_max)


def run_scenario(teacher, seed):
    scenario = mk_scenario_from_teacher(teacher, seed)
    scenario.run(max_events=MAX_EVENTS)
    return scenario