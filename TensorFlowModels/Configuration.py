def configuration(name, config):
    name = name.split("-")[0]
    value = config.get(name)
    if (name == "lstm"):
        n_input = value.get('n_input')
        n_nodes = value.get('n_nodes')
        n_epochs = value.get('n_epochs')
        n_batch = value.get('n_batch')
        n_diff = value.get('n_diff')
        n_features = value.get('n_features')
        configs = list()
        for i in n_epochs:
            for j in n_batch:
                for k in n_diff:
                    for l in n_features:
                        for m in n_input:
                            for n in n_nodes:
                                cfg = {'n_input': m,
                                       'n_nodes': n,
                                       'n_epochs': i,
                                       'n_batch': j,
                                       'n_diff': k,
                                       'n_features': l}
                                configs.append(cfg)
        return configs

    if (name == "rnn"):
        n_input = value.get('n_input')
        n_nodes = value.get('n_nodes')
        n_epochs = value.get('n_epochs')
        n_batch = value.get('n_batch')
        n_diff = value.get('n_diff')
        n_features = value.get('n_features')
        configs = list()
        for i in n_epochs:
            for j in n_batch:
                for k in n_diff:
                    for l in n_features:
                        for m in n_input:
                            for n in n_nodes:
                                cfg = {'n_input': m,
                                       'n_nodes': n,
                                       'n_epochs': i,
                                       'n_batch': j,
                                       'n_diff': k,
                                       'n_features': l}
                                configs.append(cfg)
        return configs

    if (name == "cnn"):
        n_input = value.get('n_input')
        n_nodes = value.get('n_nodes')
        n_epochs = value.get('n_epochs')
        n_batch = value.get('n_batch')
        n_diff = value.get('n_diff')
        n_features = value.get('n_features')
        configs = list()
        for i in n_epochs:
            for j in n_batch:
                for k in n_diff:
                    for l in n_features:
                        for m in n_input:
                            for n in n_nodes:
                                cfg = {'n_input': m,
                                       'n_nodes': n,
                                       'n_epochs': i,
                                       'n_batch': j,
                                       'n_diff': k,
                                       'n_features': l}
                                configs.append(cfg)
        return configs

    if (name == "mlp"):
        n_input = value.get('n_input')
        n_nodes = value.get('n_nodes')
        n_epochs = value.get('n_epochs')
        n_batch = value.get('n_batch')
        n_diff = value.get('n_diff')
        n_features = value.get('n_features')
        configs = list()
        for i in n_epochs:
            for j in n_batch:
                for k in n_diff:
                    for l in n_features:
                        for m in n_input:
                            for n in n_nodes:
                                cfg = {'n_input': m,
                                       'n_nodes': n,
                                       'n_epochs': i,
                                       'n_batch': j,
                                       'n_diff': k,
                                       'n_features': l}
                                configs.append(cfg)
        return configs


def default_config():
    config = {'lstm':{
                'n_input':[12],
                'n_nodes':[100],
                'n_epochs':[2,3],
                'n_batch':[1],
                'n_diff':[12],
                'n_features':[1]},
        'mlp':{
                'n_input':[12],
                'n_nodes':[100],
                'n_epochs':[2,3],
                'n_batch':[1],
                'n_diff':[12],
                'n_features':[1]},
        'rnn':{
            'n_input':[12],
            'n_nodes':[70],
            'n_epochs':[2],
            'n_batch':[1],
            'n_diff':[12],
            'n_features':[1]},
        'cnn':{
            'n_input':[12],
            'n_nodes':[70],
            'n_epochs':[2],
            'n_batch':[1],
            'n_diff':[12],
            'n_features':[1]}
        }
    return config