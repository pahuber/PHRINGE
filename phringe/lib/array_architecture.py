from phringe.lib.array_configuration import XArrayConfiguration, KiteArrayConfiguration, PentagonArrayConfiguration, \
    Linear2ArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell, Kernel4, Kernel5, SingleBracewell


class LIFEReferenceArchitecture(XArrayConfiguration, DoubleBracewell):
    pass


class Kernel4Kite(KiteArrayConfiguration, Kernel4):
    pass


class Kernel5Pentagon(PentagonArrayConfiguration, Kernel5):
    pass


class SingleBracewellArray(Linear2ArrayConfiguration, SingleBracewell):
    pass
