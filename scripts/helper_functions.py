
def get_parameter_groups(model, classifier_dict):
    temporal_proc_params = []
    head_params = []

    if classifier_dict['temporal_processing'] == 'gated_pooling' and classifier_dict['head'] == 'evidential':
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("heads"):
                head_params.append(p)
            elif name.startswith("pool"):
                temporal_proc_params.append(p)
    else: raise Exception("Functionality not implemented. Only option available is 'gated_pooling' and 'evidential'")

    return temporal_proc_params, head_params