from xiaoming.context.windows import compact_threshold_tokens, model_context_window_tokens


def test_model_context_window_uses_model_specific_sizes():
    assert model_context_window_tokens("deepseek-v4-flash") == 1_000_000
    assert model_context_window_tokens("deepseek-chat") == 1_000_000
    assert model_context_window_tokens("gpt-5") == 400_000
    assert model_context_window_tokens("gpt-4.1") == 1_047_576


def test_compact_threshold_scales_with_model_window_and_output_budget():
    small = compact_threshold_tokens("unknown-model", 64_000)
    large = compact_threshold_tokens("deepseek-v4-flash", 64_000)

    assert small < large
    assert small == 115_200
    assert large == 900_000
