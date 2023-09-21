class Counter:
    def __init__(self):
        self._counters = {}

    def reset(self, key: str):
        self._counters[key] = 0

    def reset_all(self):
        self._counters = {}

    def increment(self, key: str) -> int:
        if key not in self._counters:
            self._counters[key] = 0

        self._counters[key] += 1

    def increment_and_get(self, key: str) -> int:
        self.increment(key)

        return self._counters[key]

    def get(self, key: str) -> int:
        if key not in self._counters:
            return 0

        return self._counters[key]

    def items(self):
        return self._counters.items()

    def __str__(self):
        return str(self._counters)

    def __repr__(self):
        return str(self)
