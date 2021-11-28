import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Particle:
    state: np.ndarray
    weight: float


class ParticleFilter(ABC):
    def __init__(self, x_0: np.ndarray, nb_particles: int, r_eff: float, resample_threshold: float):
        self.nb_particles = nb_particles
        self.particles = [Particle(x_0, 1 / nb_particles) for _ in range(nb_particles)]
        self.n_eff_threshold = r_eff * self.nb_particles
        self.resample_threshold = resample_threshold

    @abstractmethod
    def predict_next_step(self, p: Particle, u: np.ndarray):
        ...

    @abstractmethod
    def pdf_reading(self, p: Particle, z: np.ndarray):
        ...

    def step(self, u: np.ndarray):
        for i in range(self.nb_particles):
            self.particles[i] = self.predict_next_step(self.particles[i], u)

    def update(self, z: np.ndarray):
        total_weight = 0
        for i in range(self.nb_particles):
            p = self.particles[i]
            pdf = self.pdf_reading(p, z)
            self.particles[i].weight *= pdf

            total_weight += self.particles[i].weight
            if total_weight / self.nb_particles < self.resample_threshold:
                self.particles = self._resample(self.particles, self.nb_particles)
            else:
                for i in range(self.nb_particles):
                    self.particles[i].weight /= total_weight

            n_eff = 1 / sum(map(lambda p: p.weight ** 2, self.particles))
            if n_eff < self.n_eff_threshold:
                self.particles = self._resample(self.particles, self.nb_particles)

    @staticmethod
    def _resample(particles, nb_particles):
        states = list(map(lambda p: p.state, particles))
        weights = list(map(lambda p: p.weight, particles))
        new_states = random.choices(states, weights, k=nb_particles)
        new_weight = 1 / nb_particles
        return list(map(lambda s: Particle(s, new_weight), new_states))

    @property
    def x_hat(self):
        sum_of_states = sum(map(lambda p: p.state, self.particles))
        return sum_of_states / self.nb_particles


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    class TestPF(ParticleFilter):
        def __init__(self, x_0: np.ndarray, nb_particles: int, r_eff: float, resample_threshold: float,
                     step_std: float, update_std: float):
            super().__init__(x_0, nb_particles, r_eff, resample_threshold)
            self.step_std = step_std
            self.update_std = update_std

        def predict_next_step(self, p: Particle, u: np.ndarray):
            new_state = p.state + u + np.random.randn() * self.step_std
            return Particle(new_state, p.weight)

        def pdf_reading(self, p: Particle, z: np.ndarray):
            dist = stats.norm(loc=p.state, scale=self.update_std)
            return dist.pdf(z)


    x_0 = np.array([[0]])
    nb_particles = 100
    r_eff = 50
    resample_threshold = 1e-5

    true_position = 0
    nb_steps = 100
    step_size = 1
    step_std = 2
    update_std = 25

    pf = TestPF(x_0, nb_particles, r_eff, resample_threshold, step_std, update_std)

    true_positions = []
    measurements = []
    estimates = []

    for i in range(nb_steps):
        true_position += step_size + np.random.randn() * step_std
        measurement = true_position + np.random.randn() * update_std

        pf.step(np.array([step_size]))
        pf.update(measurement)

        true_positions.append(true_position)
        measurements.append(measurement)
        estimates.append(pf.x_hat)


    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    m_average = moving_average(measurements, n=5).squeeze()
    estimates = np.array(estimates).squeeze()

    plt.plot(range(nb_steps), true_positions, label='true_positions')
    plt.scatter(range(nb_steps), measurements, marker='x', label='measurements')
    plt.plot(range(nb_steps), estimates, label='estimates')
    plt.plot(range(len(m_average)), m_average, label='moving average')
    plt.title('Particle filter')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend()
    plt.show()
