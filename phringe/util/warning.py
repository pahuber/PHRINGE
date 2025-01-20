class MissingRequirementWarning(Warning):
    """
    A custom warning indicating that a certain attribute or object
    is required to perform a particular calculation.
    """

    def __init__(self, missing_attribute: str, derived_attribute: str):
        # We build the message using f-string interpolation:
        message = f"{missing_attribute} is required to calculate {derived_attribute}."
        super().__init__(message)
