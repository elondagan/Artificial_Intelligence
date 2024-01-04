import bisect


class PriorityQueue:
    def __init__(self, order=min, f=lambda x: x):
        self.queue = []
        self.order = order
        self.f = f

    def push(self, item):
        bisect.insort(self.queue, (self.f(item), item))

    def pop(self):
        return self.queue.pop(0)[1]

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return any(item == tup[1] for tup in self.queue)

    def __getitem__(self, key):
        for _, item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (_, k) in enumerate(self.queue):
            if k == key:
                self.queue.pop(i)

    def __repr__(self):
        return str(self.queue)


class LazyAstarQueue:

    def __init__(self):
        self.queue = []

    def push(self, node):
        bisect.insort(self.queue, (node.fs, node))

    def pop(self):
        return self.queue.pop(0)[1]

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return any(item == tup[1] for tup in self.queue)

    def __getitem__(self, key):
        for _, item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (_, k) in enumerate(self.queue):
            if k == key:
                self.queue.pop(i)

    def __repr__(self):
        return str(self.queue)


class PredictiveAstarQueue:

    def __init__(self):
        self.queue = []

    def push(self, node):
        bisect.insort(self.queue, (node.fs, node.order, node))

    def pop(self):
        return self.queue.pop(0)[2]

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return any(item == tup[2] for tup in self.queue)

    def __getitem__(self, key):
        for _, _, item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (_, _, k) in enumerate(self.queue):
            if k == key:
                self.queue.pop(i)

    def __repr__(self):
        return str(self.queue)
