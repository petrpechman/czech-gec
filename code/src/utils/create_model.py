from transformers import BartConfig
from transformers import TFAutoModelForSeq2SeqLM

config = BartConfig(
                vocab_size=32_000,
                max_position_embeddings=256,
                encoder_layers=6,
                encoder_ffn_dim=2048,
                encoder_attention_heads=8,
                decoder_layers=6,
                decoder_ffn_dim=2048,
                decoder_attention_heads=8,
                encoder_layerdrop=0.0,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=512,
                dropout=0.1,
                attention_dropout=0.0,
                activation_dropout=0.0,
                init_std=0.02,
                classifier_dropout=0.0,
                scale_embedding=True,
                use_cache=True,
                num_labels=3,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                is_encoder_decoder=True,
                decoder_start_token_id=1,
                forced_eos_token_id=2,
            )

model = TFAutoModelForSeq2SeqLM.from_config(config)
model.save_pretrained("../../models/transformer/")