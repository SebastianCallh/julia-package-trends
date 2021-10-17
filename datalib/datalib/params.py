import yaml
from .load import root_path

def priors():
    with open(root_path() / "params" / "priors.yaml", "r") as fd:
        return yaml.safe_load(fd)
