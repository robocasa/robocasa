from robosuite.utils.errors import robosuiteError


class PlacementError(robosuiteError):
    """Exception raised for errors related to object placement."""

    pass


class SamplingError(robosuiteError):
    """Exception raised for errors related to sampling."""

    pass
