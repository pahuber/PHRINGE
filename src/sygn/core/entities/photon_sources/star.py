from pydantic import BaseModel


class Star(BaseModel):
  name: str
  distance: str
  mass: str
  radius: str
  temperature: str
  luminosity: str
  right_ascension: str
  declination: str