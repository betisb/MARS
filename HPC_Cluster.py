import math

class Machine:
    def __init__(self, id):
        self.id = id
        self.running_task_id = -1
        self.is_free = True
        self.task_history = []
    def taken_by_task(self, task_id):
        if self.is_free:
            self.running_task_id = task_id
            self.is_free = False
            self.task_history.append(task_id)
            return True
        else:
            return False
    def reset(self):
        self.is_free = True
        self.running_task_id = -1
        self.task_history = []
    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_task_id = -1
            return 1



    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

        for i in range(self.total_node):
            self.all_nodes.append(Machine(i))

    def feature(self):
        return [self.free_node]

    def can_allocated(self, task):
        if task.request_number_of_nodes != -1 and task.request_number_of_nodes > self.free_node:
            return False
        if task.request_number_of_nodes != -1 and task.request_number_of_nodes <= self.free_node:
            return True

    def allocate(self, task_id, request_num_procs):
        allocated_nodes = []
        request_node = int(math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_task(task_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print ("Error in allocation, there are enough free resources but can not allocated!")
        return []


        request_node = int(math.ceil(float(task.request_number_of_processors)/float(self.num_procs_per_node)))
        task.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True


    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        for m in self.all_nodes:
            m.reset()


    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()