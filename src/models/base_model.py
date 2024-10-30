# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tensorflow as tf
import numpy as np
import json
from abc import abstractmethod
from src.helper import data_ploter
from src.helper.data_logger import generate_algorithm_logger


class BaseTFModel(object):

    def __init__(self, env, **options):
        self.env = env
        self.total_step = 0

        # Handle options with defaults
        self.learning_rate = options.get('learning_rate', 0.001)
        self.batch_size = options.get('batch_size', 32)
        self.logger = options.get('logger', generate_algorithm_logger('model'))
        self.enable_saver = options.get("enable_saver", False)
        self.enable_summary_writer = options.get('enable_summary_writer', False)
        self.save_path = options.get("save_path", None)
        self.summary_path = options.get("summary_path", None)
        self.mode = options.get('mode', 'train')

        # Initialize saver and summary writer if enabled
        self._init_saver()
        self._init_summary_writer()

    def restore(self):
        if self.enable_saver and self.save_path:
            self.checkpoint.restore(self.save_path)

    def _init_saver(self):
        if self.enable_saver:
            self.checkpoint = tf.train.Checkpoint(model=self)

    def _init_summary_writer(self):
        if self.enable_summary_writer:
            self.summary_writer = tf.summary.create_file_writer(self.summary_path)

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        return None, None, None

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def add_rnn(layer_count, hidden_size, cell=tf.keras.layers.LSTMCell, activation=tf.keras.activations.tanh):
        """Creates a multi-layer RNN with LSTM cells."""
        cells = [cell(hidden_size, activation=activation) for _ in range(layer_count)]
        return tf.keras.layers.RNN(cells)

    @staticmethod
    def add_cnn(x_input, filters, kernel_size, pooling_size):
        """Creates a CNN block with Conv2D and MaxPooling2D layers."""
        convoluted_tensor = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME', activation='relu')(x_input)
        return tf.keras.layers.MaxPooling2D(pooling_size, strides=(1, 1), padding='SAME')(convoluted_tensor)

    @staticmethod
    def add_fc(x, units, activation=None):
        """Creates a fully connected (dense) layer."""
        return tf.keras.layers.Dense(units, activation=activation)(x)


class BaseRLTFModel(BaseTFModel):

    def __init__(self, env, a_space, s_space, **options):
        super().__init__(env, **options)
        self.a_space, self.s_space = a_space, s_space

        # Handle options with defaults
        self.episodes = options.get('episodes', 30)
        self.gamma = options.get('gamma', 0.9)
        self.tau = options.get('tau', 0.01)
        self.epsilon = options.get('epsilon', 0.9)
        self.buffer_size = options.get('buffer_size', 10000)
        self.save_episode = options.get("save_episode", 10)

    def eval(self):
        self.mode = 'test'
        s = self.env.reset('eval')
        while True:
            c, a, _ = self.predict(s)
            s_next, r, status, info = self.env.forward(c, a)
            s = s_next
            if status == self.env.Done:
                self.env.trader.log_asset(0)
                break

    def plot(self):
        with open(self.save_path + '_history_profits.json', 'w') as fp:
            json.dump(self.env.trader.history_profits, fp, indent=True)

        with open(self.save_path + '_baseline_profits.json', 'w') as fp:
            json.dump(self.env.trader.history_baselines, fp, indent=True)

        data_ploter.plot_profits_series(
            self.env.trader.history_baselines,
            self.env.trader.history_profits,
            self.save_path
        )

    def save(self, episode):
        self.checkpoint.save(self.save_path)
        self.logger.warning(f"Episode: {episode} | Saver reached checkpoint.")

    @abstractmethod
    def save_transition(self, s, a, r, s_next):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        """Convert action values to indices."""
        return np.where(a > 1 / 3, 2, np.where(a < -1 / 3, 1, 0)).astype(np.int32).tolist()

    def get_stock_code_and_action(self, a, use_greedy=False, use_prob=False):
        """Calculate stock code and action index."""
        if not use_greedy:
            a = a.flatten()
            action_index = np.random.choice(np.arange(a.size), p=a) if use_prob else np.argmax(a)
        else:
            action_index = np.floor(a).astype(int) if np.random.uniform() < self.epsilon else np.random.randint(0, self.a_space)

        action = action_index % 3
        stock_index = action_index // 3
        stock_code = self.env.codes[stock_index]

        return stock_code, action, action_index



class BaseSLTFModel(BaseTFModel):

    def __init__(self, session, env, **options):
        super(BaseSLTFModel, self).__init__(session, env, **options)

        # Initialize parameters.
        self.x, self.label, self.y, self.loss = None, None, None, None

        try:
            self.train_steps = options["train_steps"]
        except KeyError:
            self.train_steps = 30000

        try:
            self.save_step = options["save_step"]
        except KeyError:
            self.save_step = 1000

    def run(self):
        if self.mode == 'train':
            self.train()
        else:
            self.restore()

    def save(self, step):
        self.saver.save(self.session, self.save_path)
        self.logger.warning("Step: {} | Saver reach checkpoint.".format(step + 1))

    def eval_and_plot(self):

        x, label = self.env.get_test_data()

        y = self.predict(x)

        with open(self.save_path + '_y.json', mode='w') as fp:
            json.dump(y.tolist(), fp, indent=True)

        with open(self.save_path + '_label.json', mode='w') as fp:
            json.dump(label.tolist(), fp, indent=True)

        data_ploter.plot_stock_series(self.env.codes,
                                      y,
                                      label,
                                      self.save_path)


class BasePTModel(object):

    def __init__(self, env, **options):

        self.env = env

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        pass

    @abstractmethod
    def restore(self):
        pass

    @abstractmethod
    def run(self):
        pass


class BaseRLPTModel(BasePTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(BaseRLPTModel, self).__init__(env, **options)

        self.env = env

        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def save_transition(self, s, a, r, s_n):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 2, np.where(a < - 1 / 3, 1, 0)).astype(np.int32)[0].tolist()
        return a
# print("base_model.py executeed successfully")