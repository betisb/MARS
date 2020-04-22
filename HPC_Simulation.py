from HPC_Task import Task, Workloads
from HPC_Cluster import Cluster

import os
import math
import json
import time
import sys
import random
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import scipy.signal

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding




#define MAX queue Size
MAX_QUEUE_SIZE = 512
#define MLP Size
MLP_SIZE = 1024
MAX_WAIT_TIME = 8 * 60 * 60  # wait time is 8 hours.
MAX_RUN_TIME = 8 * 60 * 60  # runtime is 8 hours
# each task has three features: wait_time, cost , runtime, machine states,
TASK_FEATURES = 4
DEBUG = False
TASK_SEQUENCE_SIZE = 512



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class HPC_Environment(gym.Env):
    def __init__(self):
        super(HPC_Environment, self).__init__()
        print("Initialize")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(TASK_FEATURES * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)

        self.task_queue = []
        self.running_tasks = []
        self.visible_tasks = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_task_idx = 0
        self.last_task_in_batch = 0
        self.num_task_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_task = False
        self.scheduled_scores = []

        self.enable_preworkloads = False
        self.pre_workloads = []

    def my_init(self, workload_file='', sched_file=''):
        print("loading from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)
        self.penalty_task_score = TASK_SEQUENCE_SIZE * self.loads.max_exec_time / 10

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f1_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        return (np.log10(request_time) * request_processors + 870 * np.log10(submit_time))

    def f2_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, task):
        request_time = task.request_time
        submit_time = task.submit_time
        return (request_time, submit_time)

    def smallest_score(self, task):
        request_processors = task.request_number_of_processors
        submit_time = task.submit_time
        return (request_processors, submit_time)

    def wfp_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        waiting_time = task.scheduled_time - task.submit_time
        return -np.power(float(waiting_time) / request_time, 3) * request_processors

    def uni_score(self, task):
        submit_time = task.submit_time
        request_processors = task.request_number_of_processors
        request_time = task.request_time
        waiting_time = task.scheduled_time - task.submit_time
        return -(waiting_time + 1e-15) / (np.log2(request_processors + 1e-15) * request_time)

    def fcfs_score(self, task):
        submit_time = task.submit_time
        return submit_time

    def gen_preworkloads(self, size):

        running_task_size = size
        for i in range(running_task_size):
            _task = self.loads[self.start - i - 1]
            req_num_of_processors = _task.request_number_of_processors
            runtime_of_task = _task.request_time
            task_tmp = Task()
            task_tmp.task_id = (-1 - i)
            task_tmp.request_number_of_processors = req_num_of_processors
            task_tmp.run_time = runtime_of_task
            if self.cluster.can_allocated(task_tmp):
                self.running_tasks.append(task_tmp)
                task_tmp.scheduled_time = max(0, (self.current_timestamp - random.randint(0, max(runtime_of_task, 1))))
                task_tmp.allocated_machines = self.cluster.allocate(task_tmp.task_id, task_tmp.request_number_of_processors)
                self.pre_workloads.append(task_tmp)
            else:
                break

    def refill_preworkloads(self):
        for _task in self.pre_workloads:
            self.running_tasks.append(_task)
            _task.allocated_machines = self.cluster.allocate(_task.task_id, _task.request_number_of_processors)

    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.task_queue = []
        self.running_tasks = []
        self.visible_tasks = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_task_idx = 0
        self.last_task_in_batch = 0
        self.num_task_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_task = False
        self.scheduled_scores = []

        task_sequence_size = TASK_SEQUENCE_SIZE

        self.pre_workloads = []

        self.start = self.np_random.randint(task_sequence_size, (self.loads.size() - task_sequence_size - 1))
        self.start_idx_last_reset = self.start
        self.num_task_in_batch = task_sequence_size
        self.last_task_in_batch = self.start + self.num_task_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.task_queue.append(self.loads[self.start])
        self.next_arriving_task_idx = self.start + 1

        if self.enable_preworkloads:
            self.gen_preworkloads(task_sequence_size + self.np_random.randint(task_sequence_size))

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.smallest_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.fcfs_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f1_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f2_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f3_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f4_score).values()))

        return self.build_observation(), self.build_critic_observation()



    def reset_for_test(self, num, start):
        self.cluster.reset()
        self.loads.reset()

        self.task_queue = []
        self.running_tasks = []
        self.visible_tasks = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_task_idx = 0
        self.last_task_in_batch = 0
        self.num_task_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_task = False
        self.scheduled_scores = []

        task_sequence_size = num

        self.start = self.np_random.randint(task_sequence_size, (self.loads.size() - task_sequence_size - 1))
        self.start_idx_last_reset = self.start
        self.num_task_in_batch = task_sequence_size
        self.last_task_in_batch = self.start + self.num_task_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.task_queue.append(self.loads[self.start])
        self.next_arriving_task_idx = self.start + 1

    def moveforward_for_resources_backfill_greedy(self, task, scheduled_logs):
        assert not self.cluster.can_allocated(task)

        earliest_start_time = self.current_timestamp
        self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_task in self.running_tasks:
            free_processors += len(running_task.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_task.scheduled_time + running_task.request_time)
            if free_processors >= task.request_number_of_processors:
                break

        while not self.cluster.can_allocated(task):

            #backfill tasks
            self.task_queue.sort(key=lambda _j: self.fcfs_score(_j))
            task_queue_iter_copy = list(self.task_queue)
            for _j in task_queue_iter_copy:
                if self.cluster.can_allocated(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    assert _j.scheduled_time == -1
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.task_id, _j.request_number_of_processors)
                    self.running_tasks.append(_j)
                    score = (self.task_score(_j) / self.num_task_in_batch)
                    scheduled_logs[_j.task_id] = score
                    self.task_queue.remove(_j)

            assert self.running_tasks
            self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.run_time))
            next_resource_release_time = (self.running_tasks[0].scheduled_time + self.running_tasks[0].run_time)
            next_resource_release_machines = self.running_tasks[0].allocated_machines

            if self.next_arriving_task_idx < self.last_task_in_batch \
                    and self.loads[self.next_arriving_task_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_task_idx].submit_time)
                self.task_queue.append(self.loads[self.next_arriving_task_idx])
                self.next_arriving_task_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_tasks.pop(0)

    def schedule_curr_sequence_reset(self, score_fn):
        scheduled_logs = {}
        while True:
            self.task_queue.sort(key=lambda j: score_fn(j))
            task_for_scheduling = self.task_queue[0]
            if not self.cluster.can_allocated(task_for_scheduling):
                self.moveforward_for_resources_backfill_greedy(task_for_scheduling, scheduled_logs)

            assert task_for_scheduling.scheduled_time == -1
            task_for_scheduling.scheduled_time = self.current_timestamp
            task_for_scheduling.allocated_machines = self.cluster.allocate(task_for_scheduling.task_id,
                                                                          task_for_scheduling.request_number_of_processors)
            self.running_tasks.append(task_for_scheduling)
            score = (self.task_score(task_for_scheduling) / self.num_task_in_batch)
            scheduled_logs[task_for_scheduling.task_id] = score
            self.task_queue.remove(task_for_scheduling)

            not_empty = self.moveforward_for_task()
            if not not_empty:
                break

        self.cluster.reset()
        self.loads.reset()
        self.task_queue = []
        self.running_tasks = []
        self.visible_tasks = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.task_queue.append(self.loads[self.start])
        self.last_task_in_batch = self.start + self.num_task_in_batch
        self.next_arriving_task_idx = self.start + 1

        if self.enable_preworkloads:
            self.refill_preworkloads()

        return scheduled_logs

    def build_critic_observation(self):
        vector = np.zeros(TASK_SEQUENCE_SIZE * 3, dtype=float)
        earlist_task = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_task.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_task_in_batch + 1):
            task = self.loads[i]
            submit_time = task.submit_time - earlist_submit_time
            request_processors = task.request_number_of_processors
            request_time = task.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            pairs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(TASK_SEQUENCE_SIZE):
            vector[i * 3:(i + 1) * 3] = pairs[i]

        return vector

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE) * TASK_FEATURES, dtype=float)
        self.task_queue.sort(key=lambda task: self.fcfs_score(task))
        self.visible_tasks = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.task_queue):
                self.visible_tasks.append(self.task_queue[i])
            else:
                break
        self.visible_tasks.sort(key=lambda j: self.fcfs_score(j))

        self.visible_tasks = []
        if len(self.task_queue) <= MAX_QUEUE_SIZE:
            for i in range(0, len(self.task_queue)):
                self.visible_tasks.append(self.task_queue[i])
        else:
            visible_f1 = []
            f1_index = 0
            self.task_queue.sort(key=lambda task: self.f1_score(task))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f1.append(self.task_queue[i])

            visible_f2 = []
            f2_index = 0
            self.task_queue.sort(key=lambda task: self.f2_score(task))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f2.append(self.task_queue[i])

            visible_sjf = []
            sjf_index = 0
            self.task_queue.sort(key=lambda task: self.sjf_score(task))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_sjf.append(self.task_queue[i])

            visible_small = []
            small_index = 0
            self.task_queue.sort(key=lambda task: self.smallest_score(task))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_small.append(self.task_queue[i])

            visible_random = []
            random_index = 0
            shuffled = list(self.task_queue)
            shuffle(shuffled)
            for i in range(0, MAX_QUEUE_SIZE):
                visible_random.append(shuffled[i])

            index = 0

            while index < MAX_QUEUE_SIZE:
                f1_task = visible_f1[f1_index]
                f1_index += 1
                f2_task = visible_f2[f2_index]
                f2_index += 1
                sjf_task = visible_sjf[sjf_index]
                sjf_index += 1
                small_task = visible_small[small_index]
                small_index += 1
                random_task = visible_sjf[random_index]
                random_index += 1

                if (not sjf_task in self.visible_tasks) and index < MAX_QUEUE_SIZE:
                    self.visible_tasks.append(sjf_task)
                    index += 1
                if (not small_task in self.visible_tasks) and index < MAX_QUEUE_SIZE:
                    self.visible_tasks.append(small_task)
                    index += 1
                if (not random_task in self.visible_tasks) and index < MAX_QUEUE_SIZE:
                    self.visible_tasks.append(random_task)
                    index += 1

        self.pairs = []
        add_skip = False
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_tasks) and i < (MAX_QUEUE_SIZE):
                task = self.visible_tasks[i]
                submit_time = task.submit_time
                request_processors = task.request_number_of_processors
                request_time = task.request_time
                wait_time = self.current_timestamp - submit_time

                normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
                normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

                if self.cluster.can_allocated(task):
                    can_schedule_now = 1.0 - 1e-5
                else:
                    can_schedule_now = 1e-5
                self.pairs.append(
                    [task, normalized_wait_time, normalized_run_time, normalized_request_nodes, can_schedule_now])
            else:
                self.pairs.append([None, 0, 1, 1, 0])

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i * TASK_FEATURES:(i + 1) * TASK_FEATURES] = self.pairs[i][1:]

        return vector

    def moveforward_for_resources_backfill(self, task):
        assert not self.cluster.can_allocated(task)
        earliest_start_time = self.current_timestamp
        self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_task in self.running_tasks:
            free_processors += len(running_task.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_task.scheduled_time + running_task.request_time)
            if free_processors >= task.request_number_of_processors:
                break

        while not self.cluster.can_allocated(task):
            self.task_queue.sort(key=lambda _j: self.fcfs_score(_j))
            task_queue_iter_copy = list(self.task_queue)
            for _j in task_queue_iter_copy:
                if self.cluster.can_allocated(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    assert _j.scheduled_time == -1
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.task_id, _j.request_number_of_processors)
                    self.running_tasks.append(_j)
                    score = (self.task_score(_j) / self.num_task_in_batch)
                    self.scheduled_rl[_j.task_id] = score
                    self.task_queue.remove(_j)
            assert self.running_tasks
            self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.run_time))
            next_resource_release_time = (self.running_tasks[0].scheduled_time + self.running_tasks[0].run_time)
            next_resource_release_machines = self.running_tasks[0].allocated_machines

            if self.next_arriving_task_idx < self.last_task_in_batch \
                    and self.loads[self.next_arriving_task_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_task_idx].submit_time)
                self.task_queue.append(self.loads[self.next_arriving_task_idx])
                self.next_arriving_task_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_tasks.pop(0)

    def skip_for_resources(self):
        assert self.running_tasks
        self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.run_time))
        next_resource_release_time = (self.running_tasks[0].scheduled_time + self.running_tasks[0].run_time)
        next_resource_release_machines = self.running_tasks[0].allocated_machines

        if self.next_arriving_task_idx < self.last_task_in_batch and self.loads[
            self.next_arriving_task_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_task_idx].submit_time)
            self.task_queue.append(self.loads[self.next_arriving_task_idx])
            self.next_arriving_task_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_tasks.pop(0)
        return False

    def moveforward_for_task(self):
        if self.task_queue:
            return True
        if self.next_arriving_task_idx >= self.last_task_in_batch:
            assert not self.task_queue
            return False
        while not self.task_queue:
            if not self.running_tasks:
                next_resource_release_time = sys.maxsize
                next_resource_release_machines = []
            else:
                self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.run_time))
                next_resource_release_time = (self.running_tasks[0].scheduled_time + self.running_tasks[0].run_time)
                next_resource_release_machines = self.running_tasks[0].allocated_machines

            if self.loads[self.next_arriving_task_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_task_idx].submit_time)
                self.task_queue.append(self.loads[self.next_arriving_task_idx])
                self.next_arriving_task_idx += 1
                return True
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_tasks.pop(0)

    def task_score(self, task_for_scheduling):
        COST = math.log(task_for_scheduling.run_time)
        _tmp = COST * max(1.0, (float(
            task_for_scheduling.scheduled_time - task_for_scheduling.submit_time + task_for_scheduling.run_time)
                         /
                         max(task_for_scheduling.run_time, 10)))
        return _tmp

    def has_only_one_task(self):
        if len(self.task_queue) == 1:
            return True
        else:
            return False

    def skip_schedule(self):
        next_resource_release_time = sys.maxsize
        next_resource_release_machines = []
        if self.running_tasks:
            self.running_tasks.sort(key=lambda running_task: (running_task.scheduled_time + running_task.run_time))
            next_resource_release_time = (self.running_tasks[0].scheduled_time + self.running_tasks[0].run_time)
            next_resource_release_machines = self.running_tasks[0].allocated_machines

        if self.next_arriving_task_idx >= self.last_task_in_batch and not self.running_tasks:
            if not self.pivot_task:
                self.pivot_task = True
                return False, 0
            else:
                return False, 0

        if self.next_arriving_task_idx < self.last_task_in_batch and self.loads[
            self.next_arriving_task_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_task_idx].submit_time)
            self.task_queue.append(self.loads[self.next_arriving_task_idx])
            self.next_arriving_task_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_tasks.pop(0)
        return False, 0

    def schedule(self, task_for_scheduling):
        if not self.cluster.can_allocated(task_for_scheduling):
            self.moveforward_for_resources_backfill(task_for_scheduling)

        assert task_for_scheduling.scheduled_time == -1
        task_for_scheduling.scheduled_time = self.current_timestamp
        task_for_scheduling.allocated_machines = self.cluster.allocate(task_for_scheduling.task_id,
                                                                      task_for_scheduling.request_number_of_processors)
        self.running_tasks.append(task_for_scheduling)
        score = (self.task_score(task_for_scheduling) / self.num_task_in_batch)
        self.scheduled_rl[task_for_scheduling.task_id] = score
        self.task_queue.remove(task_for_scheduling)

        not_empty = self.moveforward_for_task()

        if not_empty:
            return False
        else:
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]

    def step(self, a):
        task_for_scheduling = self.pairs[a][0]

        if not task_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            task_for_scheduling = self.pairs[a][0]
            done = self.schedule(task_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0]
        else:
            rl_total = sum(self.scheduled_rl.values())
            best_total = min(self.scheduled_scores)
            rwd2 = (best_total - rl_total)
            rwd = -rl_total
            return [None, rwd, True, rwd2]

    def step_for_test(self, a):
        task_for_scheduling = self.pairs[a][0]

        if not task_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            task_for_scheduling = self.pairs[a][0]
            done = self.schedule(task_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./Dataset/synthetic_small.swf')
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPC_Environment()
    env.my_init(workload_file=workload_file, sched_file=workload_file)
    env.seed(0)

    for _ in range(100):
        _, r = env.reset(), 0
        while True:
            _, r, d, _ = env.step(0)
            if d:
                print("HPC Reward:", r)
                break
