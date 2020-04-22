import re
import sys
import math


class Task:
    def __init__(self, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.task_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])

        self.request_number_of_processors = int(s_array[7])
        self.number_of_allocated_processors = max(self.number_of_allocated_processors,
                                                  self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors

        self.request_number_of_nodes = -1

        self.request_time = int(s_array[8])
        if self.request_time == -1:
            self.request_time = self.run_time


        self.request_memory = int(s_array[9])
        self.status = int(s_array[10])
        self.user_id = int(s_array[11])
        self.group_id = int(s_array[12])
        self.executable_number = int(s_array[13])
        self.queue_number = int(s_array[14])

        try:
            self.partition_number = int(s_array[15])
        except ValueError:
            self.partition_number = 0

        self.proceeding_task_number = int(s_array[16])
        self.think_time_from_proceeding_task = int(s_array[17])

        self.random_id = self.submit_time

        self.scheduled_time = -1

        self.allocated_machines = None

        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_task_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __str__(self):
        return "t[" + str(self.task_id) + "]-[" + str(self.request_number_of_processors) + "]-[" + str(
            self.submit_time) + "]-[" + str(self.request_time) + "]"

    def __feature__(self):
        return [self.submit_time, self.request_number_of_processors, self.request_time,
                self.user_id, self.group_id, self.executable_number, self.queue_number]


class Workloads:
    all_tasks = []

    def __init__(self, path):
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_task_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_task_id = 0
        self.max_nodes = 0
        self.max_procs = 0

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                t = Task(line)
                if t.run_time > self.max_exec_time:
                    self.max_exec_time = t.run_time
                if t.run_time < self.min_exec_time:
                    self.min_exec_time = t.run_time

                self.all_tasks.append(t)

                if t.request_number_of_processors > self.max:
                    self.max = t.request_number_of_processors

        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print("Max Allocated Processors:", str(self.max), ";max node:", self.max_nodes,
              ";max procs:", self.max_procs,
              ";max execution time:", self.max_exec_time)

        self.all_tasks.sort(key=lambda task: task.task_id)

    def size(self):
        return len(self.all_tasks)

    def reset(self):
        for task in self.all_tasks:
            task.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_tasks[item]


if __name__ == "__main__":
    print("Loading the workloads...")
    load = Workloads("../../../Dataset/synthetic_small.swf")
    print("Loading ...", type(load[0]))
    print(load.max_nodes, load.max_procs)
    print(load[0].__feature__())
    print(load[1].__feature__())
