from voice_clone_check.text_metrics import (
    character_error_rate,
    edit_distance,
    normalize_japanese_reading,
)


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3


def test_japanese_normalization_ignores_punctuation_and_script():
    assert normalize_japanese_reading("テスト、です！") == normalize_japanese_reading(
        "てすとです"
    )


def test_character_error_rate():
    assert character_error_rate("今日は晴れです。", "今日は晴れです") == 0
    assert character_error_rate("あいうえお", "あいうえ") == 0.2

