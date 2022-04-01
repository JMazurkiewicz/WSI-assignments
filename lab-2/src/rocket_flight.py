# Author: Jakub Mazurkiewicz
import numpy.random as npr

FLIGHT_DURATION = 200
REQUIRED_HEIGHT = 750
EMPTY_ROCKET_MASS = 20
GRAVITY = -0.9
TIME_UNIT = 1.0

def create_random_rocket_flights(pop_size, prob_1=0.5):
    return [npr.binomial(n=1, p=prob_1, size=[FLIGHT_DURATION]) for _ in range(pop_size)]

def _calc_rocket_acceleration(mass):
    return 500 / mass

def _calc_friction_acceleration(velocity, mass):
    return (-0.06 * velocity * abs(velocity)) / mass

def calc_rocket_flight_fitness(flight):
    assert len(flight) == FLIGHT_DURATION

    height = 0.0
    velocity = 0.0
    times_engine_on = sum(flight)
    mass = EMPTY_ROCKET_MASS + times_engine_on
    score = -times_engine_on

    for step in flight:
        mass -= step
        acceleration = _calc_rocket_acceleration(mass) + _calc_friction_acceleration(velocity, mass) + GRAVITY
        velocity += acceleration * TIME_UNIT
        height += (velocity * TIME_UNIT) + (acceleration * TIME_UNIT ** 2) / 2

    if height >= REQUIRED_HEIGHT:
        score += 200
    return score
