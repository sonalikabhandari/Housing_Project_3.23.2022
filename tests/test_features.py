from regression_model.config.core import config
from regression_model.processing.features import bed_bath_transformer



def test_temporal_variable_transformer(sample_input_data):
    # Given

    # clean_test_data = clean_test_data[config.model_config.features]
    # transformer = bed_bath_transformer(config.model_config.cat_vars_to_num,
    # config.model_config.variables_to_drop)
    input_data = sample_input_data[config.model_config.features]
    print(input_data)
    transformer = bed_bath_transformer(['bathroom','bed'],['county','zipcode'])
    assert input_data["bed"].iat[1] == '2 bed'

    # When

    # data = get_data(sample_input_data)
    subject = transformer.fit_transform(input_data)

    # Then
    assert subject["bed"].iat[1] == 2
