import numpy as np
import torch
import logging
import sobol_seq
import torch.multiprocessing as mp
from scipy.stats import norm
from rqmc_distributions import Normal_RQMC


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('MAIN')
logger.setLevel(logging.DEBUG)

class Normalizer:
    def __init__(self, filter_mean=True):
        self.m = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def state_dict(self):
        return {'m': self.m,
                'v': self.v,
                'n': self.n}

    def load_state_dict(self, saved):
        self.m = saved['m']
        self.v = saved['v']
        self.n = saved['n']

    def __call__(self, o):
        self.m = self.m * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o_ = (o - self.m) / self.std
        else:
            o_ = o / self.std
        return o_

class StaticNormalizer:
    def __init__(self, o_size):
        self.offline_stats = SharedStats(o_size)
        self.online_stats = SharedStats(o_size)

    def __call__(self, o_):
        if np.isscalar(o_):
            o = torch.FloatTensor([o_])
        else:
            o = torch.FloatTensor(o_)
        self.online_stats.feed(o)
        if self.offline_stats.n[0] == 0:
            return o_
        std = (self.offline_stats.v + 1e-6) ** .5
        o = (o - self.offline_stats.m) / std
        o = o.numpy()
        if np.isscalar(o_):
            o = np.asscalar(o)
        else:
            o = o.reshape(o_.shape)
        return o

class SharedStats:
    def __init__(self, o_size):
        self.m = torch.zeros(o_size)
        self.v = torch.zeros(o_size)
        self.n = torch.zeros(1)
        self.m.share_memory_()
        self.v.share_memory_()
        self.n.share_memory_()

    def feed(self, o):
        n = self.n[0]
        new_m = self.m * (n / (n + 1)) + o / (n + 1)
        self.v.copy_(self.v * (n / (n + 1)) + (o - self.m) * (o - new_m) / (n + 1))
        self.m.copy_(new_m)
        self.n.add_(1)

    def zero(self):
        self.m.zero_()
        self.v.zero_()
        self.n.zero_()

    def load(self, stats):
        self.m.copy_(stats.m)
        self.v.copy_(stats.v)
        self.n.copy_(stats.n)

    def merge(self, B):
        A = self
        n_A = self.n[0]
        n_B = B.n[0]
        n = n_A + n_B
        delta = B.m - A.m
        m = A.m + delta * n_B / n
        v = A.v * n_A + B.v * n_B + delta * delta * n_A * n_B / n
        v /= n
        self.m.copy_(m)
        self.v.copy_(v)
        self.n.add_(B.n)

    def state_dict(self):
        return {'m': self.m.numpy(),
                'v': self.v.numpy(),
                'n': self.n.numpy()}

    def load_state_dict(self, saved):
        self.m = torch.FloatTensor(saved['m'])
        self.v = torch.FloatTensor(saved['v'])
        self.n = torch.FloatTensor(saved['n'])

class Evaluator:
    def __init__(self, config, state_normalizer):
        self.model = config.model_fn()
        self.repetitions = config.repetitions
        self.env = config.env_fn()
        self.state_normalizer = state_normalizer
        self.config = config

    def eval(self, solution):
        self.model.set_weight(solution)
        rewards = []
        steps = []
        for i in range(self.repetitions):
            reward, step = self.single_run()
            rewards.append(reward)
            steps.append(step)
        return -np.mean(rewards), np.sum(steps)

    def single_run(self):
        state = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            state = self.state_normalizer(state)
            action = self.model(np.stack([state])).data.numpy().flatten()
            action += np.random.randn(len(action)) * self.config.action_noise_std
            action = self.config.action_clip(action)
            state, reward, done, info = self.env.step(action)
            steps += 1
            total_reward += reward
            if done:
                return total_reward, steps

def fitness_shift(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = self.beta2_t = 1
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

    def update(self, g):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(g, 2)
        m_ = self.m / (1 - self.beta1_t)
        v_ = self.v / (1 - self.beta2_t)
        return m_ / (np.sqrt(v_) + self.epsilon)

class LatinSupercube:
    def __init__(self, noise_sizes):
        self.noise_sizes = noise_sizes

    def sample(self, sample_size):
        noises = []
        for noise_size in self.noise_sizes:
            noise = np.random.permutation(Normal_RQMC(torch.zeros(noise_size), torch.ones(noise_size)).sample(torch.Size([sample_size])).data.numpy())
            noises.append(noise)
        noises = np.concatenate(noises, 1)
        return noises

class NoiseGenerator:
    def __init__(self, noise_size, sample_size, noise_type='mc', noise_sizes=None):
        print('noise size={}'.format(noise_size))
        self.noise_size = noise_size
        self.sample_size = sample_size
        self.noise_type = noise_type
        if self.noise_type == 'rqmc':
            self.sampler = Normal_RQMC(torch.zeros(noise_size), torch.ones(noise_size))
            self.noises = self.sampler.sample(torch.Size([sample_size])).numpy()
            self.i = 0
        elif self.noise_type == 'lss':
            self.sampler = LatinSupercube(noise_sizes)
            self.noises = self.sampler.sample(sample_size)
            self.i = 0
        self.lock = mp.Lock()

    def sample(self):
        with self.lock:
            if self.noise_type == 'mc':
                return np.random.randn(self.noise_size)
            elif self.noise_type == 'rqmc':
                noise = self.noises[self.i]
                self.i += 1
                if self.i % self.sample_size == 0:
                    self.noises = self.sampler.sample(torch.Size([self.sample_size])).numpy()
                    self.i = 0
                return noise
            elif self.noise_type == 'lss':
                noise = self.noises[self.i]
                self.i += 1
                if self.i % self.sample_size == 0:
                    self.noises = self.sampler.sample(self.sample_size)
                    self.i = 0
            else:
                raise Exception('unknown noise type')
