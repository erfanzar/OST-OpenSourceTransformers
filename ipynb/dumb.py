
d_ff = 2048,
d_kv = 64,
d_model = 768,
decoder_start_token_id = 0,
dense_act_fn = 'gelu_new',
dropout_rate = 0.1,
eos_token_id = 1,
feed_forward_proj = 'gated-gelu',
initializer_factor = 1.0,
is_encoder_decoder = True,
is_gated_act = True,
layer_norm_epsilon = 1e-06,
model_type = 't5',
n_positions = 512,
num_decoder_layers = 12,
num_heads = 12,
num_layers = 12,
output_past = True,
pad_token_id = 0,
relative_attention_max_distance = 128,
relative_attention_num_buckets = 32,
tie_word_embeddings = False,
use_cache = True,
vocab_size = 32128,
task_specific_params = {
    "summarization": {
        "early_stopping": True,
        "length_penalty": 2.0,
        "max_length": 200,
        "min_length": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "prefix": "summarize: "
    },
    "translation_en_to_de": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
        "early_stopping": True,
        "max_length": 300,
        "num_beams": 4,
        "prefix": "translate English to Romanian: "
    }
}

