from aieval.normalizers import normalize_text


def test_normalize_text():

    assert normalize_text("Paris") == "paris"
    assert normalize_text(" PARIS ") == "paris"
    assert normalize_text("Paris.") == "paris"
    assert normalize_text("  Paris   is   beautiful  ") == "paris is beautiful"