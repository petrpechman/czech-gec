
def split_features_and_labels(input_batch):
    features = {key: tensor for key, tensor in input_batch.items() if key in ['input_ids', 'attention_mask', 'decoder_input_ids']}
    labels = {key: tensor for key, tensor in input_batch.items() if key in ['labels']}
    if len(features) == 1:
        features = list(features.values())[0]
    if len(labels) == 1:
        labels = list(labels.values())[0]
    if isinstance(labels, dict) and len(labels) == 0:
        return features
    else:
        return features, labels
