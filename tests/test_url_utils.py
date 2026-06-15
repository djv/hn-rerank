from urllib.parse import urlsplit, parse_qsl

from hypothesis import assume, given, settings, strategies as st
from hypothesis.provisional import urls

from api.url_utils import normalize_url, extract_domain, _TRACKING_QUERY_PARAMS


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


# =============================================================================
# Property-based tests
# =============================================================================

_TRACKING_BARE = st.sampled_from(sorted(_TRACKING_QUERY_PARAMS))


@settings(max_examples=200, deadline=None)
@given(urls())
def test_normalize_url_idempotent(url: str) -> None:
    """normalize_url(normalize_url(u)) == normalize_url(u) for any valid URL."""
    assume("%" not in url)  # %-encoding can cause decoding→reinterpretation
    once = normalize_url(url)
    twice = normalize_url(once)
    assert once == twice, f"once={once!r} twice={twice!r}"


@settings(max_examples=200, deadline=None)
@given(
    tracking_key=_TRACKING_BARE,
    tracking_value=st.text(
        min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"
    ),
    has_other_param=st.booleans(),
)
def test_normalize_url_tracking_params_never_survive(
    tracking_key: str, tracking_value: str, has_other_param: bool
) -> None:
    """After normalization, query never contains any known tracking key."""
    base = "https://example.com/article"
    if has_other_param:
        url = f"{base}?keep=1&{tracking_key}={tracking_value}"
    else:
        url = f"{base}?{tracking_key}={tracking_value}"
    normalized = normalize_url(url)
    query = urlsplit(normalized).query
    assert tracking_key.lower() not in query.lower(), (
        f"tracking key {tracking_key!r} survived in {normalized!r}"
    )
    assert "utm_" not in query.lower()
    if has_other_param:
        assert "keep=1" in query, "non-tracking param was dropped"


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=10**9))
def test_normalize_url_hn_item_id_preserved(story_id: int) -> None:
    """For HN item URLs, the id=N query param is preserved exactly."""
    url = f"https://news.ycombinator.com/item?id={story_id}"
    normalized = normalize_url(url)
    parts = urlsplit(normalized)
    assert parts.netloc == "news.ycombinator.com", f"netloc changed: {normalized}"
    qs = dict(parse_qsl(parts.query))
    assert qs.get("id") == str(story_id), f"id param changed: {normalized}"


@settings(max_examples=100, deadline=None)
@given(urls())
def test_normalize_url_no_fragment_in_output(url: str) -> None:
    """Output URL never contains a # fragment."""
    url_with_frag = url + "#section"
    normalized = normalize_url(url_with_frag)
    assert "#" not in normalized, f"fragment present in {normalized!r}"


@settings(max_examples=200, deadline=None)
@given(
    params=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz"),
            st.text(
                min_size=1,
                max_size=8,
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            ),
        ),
        min_size=2,
        max_size=6,
        unique_by=lambda kv: kv[0],
    ),
)
def test_normalize_url_query_params_sorted(params: list[tuple[str, str]]) -> None:
    """Different orderings of the same query params produce identical output."""
    base = "https://example.com/page"
    items = [f"{k}={v}" for k, v in params]
    url_a = base + "?" + "&".join(items)
    url_b = base + "?" + "&".join(reversed(items))
    assert normalize_url(url_a) == normalize_url(url_b)


@settings(max_examples=100, deadline=None)
@given(urls())
def test_normalize_url_no_trailing_slash(url: str) -> None:
    """Output path has no trailing / (except possibly root)."""
    normalized = normalize_url(url)
    parts = urlsplit(normalized)
    assume(parts.path != "/" and bool(parts.path))
    assert not parts.path.endswith("/"), f"trailing slash in {normalized!r}"


@settings(max_examples=100, deadline=None)
@given(
    scheme=st.sampled_from(["http", "https"]),
    use_www=st.booleans(),
    domain=st.from_regex(r"[a-z][a-z0-9-]{1,30}\.[a-z]{2,4}", fullmatch=True).filter(
        lambda d: not d.startswith("www.")
    ),
)
def test_extract_domain_lowercase_strips_www(
    scheme: str, use_www: bool, domain: str
) -> None:
    """extract_domain returns lowercased netloc with 'www.' stripped."""
    host = f"www.{domain}" if use_www else domain
    url = f"{scheme}://{host}/path"
    result = extract_domain(url)
    assert result == domain, (
        f"extract_domain({url!r}) = {result!r}, expected {domain!r}"
    )


@settings(max_examples=100, deadline=None)
@given(urls())
def test_extract_domain_stable_under_normalization(url: str) -> None:
    """Domain extraction is stable when URL is normalized."""
    once = extract_domain(url)
    twice = extract_domain(normalize_url(url))
    assert once == twice, f"once={once!r} twice={twice!r}"


def test_extract_domain_empty_and_none() -> None:
    assert extract_domain("") is None
    assert extract_domain(None) is None
