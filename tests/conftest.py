import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset
from regression_model.processing.features import get_data

@pytest.fixture()
def sample_input_data():
    data =  load_dataset(file_name=config.app_config.test_data_file)
    return get_data(data)

# @pytest.fixture()
# def sample_input_data():
#     return load_dataset(file_name=config.app_config.test_data_file)
