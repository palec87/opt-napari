try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "David Palecek"
__credits__ = "Giorgia Tortora, Marcos Obando"
__email__ = "dpalecek@ualg.pt"

from ._qtwidget import PreprocessingnWidget
