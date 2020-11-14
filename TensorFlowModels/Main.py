from Functions import prepare_data, run, convert_config_to_datframes, get_config, run_model

train_X, train_y, test_X, test_y, scaler = prepare_data()

models, df_results = run(train_X, train_y, test_X, test_y, scaler)

convert_config_to_datframes(get_config(models))[0]
#
run_model(train_X, train_y, test_X,test_y, 'rnn-2', {
            'n_input':12,
            'n_nodes':70,
            'n_epochs':2,
            'n_batch':1,
            'n_diff':12,
            'n_features':1},models, df_results, scaler )