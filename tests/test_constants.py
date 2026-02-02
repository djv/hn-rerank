import pytest

from api import constants


def test_tuned_default_constants():
    """Guard tuned defaults from accidental drift."""
    assert constants.RANKING_DIVERSITY_LAMBDA == pytest.approx(0.17306440666107006)
    assert constants.RANKING_DIVERSITY_LAMBDA_CLASSIFIER == pytest.approx(
        0.17306440666107006
    )
    assert constants.RANKING_NEGATIVE_WEIGHT == pytest.approx(0.3829876085811009)
    assert constants.KNN_NEIGHBORS == 5

    assert constants.ADAPTIVE_HN_WEIGHT_MIN == pytest.approx(0.08082116780277106)
    assert constants.ADAPTIVE_HN_WEIGHT_MAX == pytest.approx(0.11427056335728537)

    assert constants.FRESHNESS_MAX_BOOST == pytest.approx(0.0022035192415596778)
    assert constants.FRESHNESS_HALF_LIFE_HOURS == pytest.approx(71.49079532287982)

    assert constants.SEMANTIC_SIGMOID_THRESHOLD == pytest.approx(0.4749411784079209)
    assert constants.SEMANTIC_SIGMOID_K == pytest.approx(31.22492938611713)
