from .base_plotter import get_plotter, register, Plotter

from . import bar_chart_time
from . import bar_chart_time_micros
from . import bar_chart_compression
from . import line_chart_compression
from . import line_chart_scalability
from . import pareto_front

__all__ = ["get_plotter", "register", "Plotter"]