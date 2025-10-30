from phringe.lib.array_configuration import XArrayConfiguration, KiteArrayConfiguration, PentagonArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell, Kernel4, Kernel5


class LIFEBaselineArchitecture(XArrayConfiguration, DoubleBracewell):
    pass


class Kernel4Kite(KiteArrayConfiguration, Kernel4):
    pass


class Kernel5Pentagon(PentagonArrayConfiguration, Kernel5):
    pass
