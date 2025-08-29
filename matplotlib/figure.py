class Axes:
    def plot(self, *args, **kwargs):
        return []

    def legend(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None


class Figure:
    def add_subplot(self, *args, **kwargs):
        ax = Axes()
        self.axes = [ax]
        return ax

    def __init__(self):
        self.axes = []

