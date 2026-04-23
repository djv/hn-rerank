from api.url_utils import normalize_url


def test_normalize_url_strips_tracking_query_fragment_and_slash():
    url = "https://Example.com/Path/?utm=1#section"
    assert normalize_url(url) == "https://example.com/Path"


def test_normalize_url_preserves_hn_item_identity():
    assert normalize_url("https://news.ycombinator.com/item?id=1") == (
        "https://news.ycombinator.com/item?id=1"
    )
    assert normalize_url("https://news.ycombinator.com/item?id=2") == (
        "https://news.ycombinator.com/item?id=2"
    )


def test_normalize_url_preserves_non_tracking_query_identity():
    assert normalize_url("https://example.com/article?id=1") == (
        "https://example.com/article?id=1"
    )
    assert normalize_url("https://example.com/article?id=2") == (
        "https://example.com/article?id=2"
    )


def test_normalize_url_sorts_query_params_deterministically():
    first = normalize_url("https://example.com/article?b=1&a=2")
    second = normalize_url("https://example.com/article?a=2&b=1")

    assert first == "https://example.com/article?a=2&b=1"
    assert second == first


def test_normalize_url_drops_tracking_params_but_keeps_meaningful_ones():
    url = "https://example.com/article?id=2&fbclid=x&utm_source=test"
    assert normalize_url(url) == "https://example.com/article?id=2"


def test_normalize_url_empty():
    assert normalize_url("") == ""
