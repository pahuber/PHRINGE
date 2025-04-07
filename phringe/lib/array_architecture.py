from phringe.lib.array_configuration import XArrayConfiguration, KiteArrayConfiguration, PentagonArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewellBeamCombiner, Kernel4BeamCombiner, Kernel5BeamCombiner


class LIFEBaselineArchitecture(XArrayConfiguration, DoubleBracewellBeamCombiner):
    pass


class Kernel4Kite(KiteArrayConfiguration, Kernel4BeamCombiner):
    pass


class Kernel5Pentagon(PentagonArrayConfiguration, Kernel5BeamCombiner):
    pass
