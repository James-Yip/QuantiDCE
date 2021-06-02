"""Provides configuration base class."""

import os
import json


class Config:
    """Base class for all configuration classes. Provides methods for loading
    or saving configurations. #TODO
    """
    def __init__(self):
        pass

    def update(self):
        """#TODO
        """
        pass

    def get_config_dict(self):
        """#TODO
        """
        pass

    def save_into_json_file(self):
        """#TODO
        """
        pass

    @classmethod
    def load_from_dict(cls):
        """#TODO
        """
        pass

    @classmethod
    def load_from_json_file(cls):
        """#TODO
        """
        pass
