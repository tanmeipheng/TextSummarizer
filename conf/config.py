class Config(object):
    # ========================================================
    # Predict Settings
    # ========================================================
    PREDICT = dict(
        model_name="facebook/bart-large-cnn",
        tokenizer_name="facebook/bart-large-cnn",
        encode_max_length=512,
        encode_truncation=True,
        generate_min_length=50,
        generate_max_length=120
        
    )