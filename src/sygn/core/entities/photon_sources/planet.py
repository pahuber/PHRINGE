from pydantic import BaseModel


class Planet(BaseModel):
    name: str
    mass: str
    radius: str
    temperature: str
    semi_major_axis: str
    eccentricity: int
    inclination: str
    raan: str
    argument_of_periapsis: str
    true_anomaly: str