from api.url_utils import normalize_url


def test_normalize_url_strips_query_fragment_and_slash():
    url = "https://Example.com/Path/?utm=1#section"
    assert normalize_url(url) == "https://example.com/Path"


def test_normalize_url_empty():
    assert normalize_url("") == ""
