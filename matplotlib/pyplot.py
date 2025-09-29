from .figure import Figure, Axes


def subplots(*args, **kwargs):
    fig = Figure()
    ax = Axes()
    fig.axes = [ax]
    return fig, ax


def plot(*args, **kwargs):
    return []


def show(*args, **kwargs):
    return None
