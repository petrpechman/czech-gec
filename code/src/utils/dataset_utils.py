import tensorflow as tf
from transformers.tf_utils import shape_list

def fix_format(input_batch, model_type):
    if model_type == "T5":
        dato = {
                "input_ids": input_batch["input_ids"],
                "attention_mask": input_batch["attention_mask"],
                "labels": input_batch["tokenized_target_line"],
                # 0 is decoder start token for T5 tokenizer
                # "decoder_input_ids": tf.concat([[0], input_batch["tokenized_target_line"][:-1]], axis=0)
            }
    elif model_type == "Bart-mine":
        dato = {
                "input_ids": input_batch["input_ids"],
                "attention_mask": input_batch["attention_mask"],
                "labels": input_batch["tokenized_target_line"][1:],
                "decoder_input_ids": input_batch["tokenized_target_line"][:-1]
            }
    return dato

def split_features_and_labels(input_batch):
    features = {key: tensor for key, tensor in input_batch.items() if key in ['input_ids', 'attention_mask', 'decoder_input_ids']}
    # features = {key: tensor for key, tensor in input_batch.items() if key in ['input_ids', 'attention_mask']}
    labels = {key: tensor for key, tensor in input_batch.items() if key in ['labels']}
    if len(features) == 1:
        features = list(features.values())[0]
    if len(labels) == 1:
        labels = list(labels.values())[0]
    if isinstance(labels, dict) and len(labels) == 0:
        return features
    else:
        return features, labels

def change_value(x, y, original_value, new_value, model_type):
    condition = tf.not_equal(y, original_value)
    changed_y = tf.where(condition, y, new_value)
    if model_type == "T5":    
        x['decoder_input_ids'] = _shift_right_t5(changed_y)
        return x, changed_y
    elif model_type == "Bart-mine":
        return x, changed_y

def _shift_right_t5(input_ids):
    # taken from https://github.com/huggingface/transformers/blob/6da93f5580e109fad5f7b523cf2b6e8a5bafb623/src/transformers/models/t5/modeling_t5.py#L880
    decoder_start_token_id = 0
    pad_token_id = 0

    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
        shifted_input_ids,
    )
    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(
        shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
    )
    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)
    return shifted_input_ids