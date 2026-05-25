import numpy as np
import pytest
from api.models import Story
from api.config import AppConfig, ClassifierConfig
from api.rerank import _classifier_metadata_features

def test_classifier_metadata_features_width():
    stories = [
        Story(
            id=1,
            title="Story 1",
            url=None,
            score=100,
            time=1000,
            text_content="Content 1",
            comment_count=50,
        ),
        Story(
            id=2,
            title="Story 2",
            url=None,
            score=200,
            time=2000,
            text_content="Content 2",
            comment_count=None,  # missing comment count test case
        )
    ]
    
    config_none = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=False,
            use_log_comments_feature=False,
            use_comment_ratio_feature=False,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=False,
        )
    )
    
    # 0 features enabled
    feats_none = _classifier_metadata_features(stories, config_none, 0.0, 2)
    assert feats_none.shape == (2, 0)
    
    # 1 feature enabled: log points
    config_points = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=True,
            use_log_comments_feature=False,
            use_comment_ratio_feature=False,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=False,
        )
    )
    feats_points = _classifier_metadata_features(stories, config_points, 0.0, 2)
    assert feats_points.shape == (2, 1)
    
    # 2 features enabled: log points + log comments
    config_comments = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=False,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=False,
        )
    )
    feats_comments = _classifier_metadata_features(stories, config_comments, 0.0, 2)
    assert feats_comments.shape == (2, 2)
    
    # 3 features enabled: log points + log comments + comment ratio
    config_all = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=True,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=False,
        )
    )
    feats_all = _classifier_metadata_features(stories, config_all, 0.0, 2)
    assert feats_all.shape == (2, 3)

def test_classifier_metadata_features_values():
    stories = [
        Story(
            id=1,
            title="Story 1",
            url=None,
            score=100,
            time=1000,
            text_content="Content 1",
            comment_count=100,
        ),
        Story(
            id=2,
            title="Story 2",
            url=None,
            score=0,
            time=2000,
            text_content="Content 2",
            comment_count=0,
        )
    ]
    
    config = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=True,
            use_log_comments_feature=True,
            use_comment_ratio_feature=True,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=False,
        )
    )
    
    feats = _classifier_metadata_features(stories, config, 0.0, 2)
    
    # Assert non-negative and properly scaled between 0 and 1.0 for points and comments
    assert np.all(feats >= 0.0)
    assert np.all(feats[:, 0] <= 1.0)  # log points normalizer cap is high, so it's < 1.0
    assert np.all(feats[:, 1] <= 1.0)  # log comments normalizer cap is 500, so 100 is < 1.0
    
    # Assert comment-to-score ratio is also non-negative and mathematically valid
    assert np.all(feats[:, 2] >= 0.0)
    
    # For Story 2: score = 0, comment_count = 0 -> ratio = log(1)/log(1)+1 = 0
    assert feats[1, 2] == pytest.approx(0.0)


def test_classifier_metadata_comments_count_uses_comment_count_field():
    stories = [
        Story(
            id=1,
            title="Story 1",
            url=None,
            score=10,
            time=1000,
            text_content="Content 1",
            comments=["a", "b", "c", "d", "e", "f", "g"],
            comment_count=2,
        )
    ]

    config = AppConfig(
        classifier=ClassifierConfig(
            use_log_points_feature=False,
            use_log_comments_feature=False,
            use_comment_ratio_feature=False,
            use_title_len_feature=False,
            use_text_len_feature=False,
            use_has_url_feature=False,
            use_github_feature=False,
            use_pdf_feature=False,
            use_comments_count_feature=True,
        )
    )

    feats = _classifier_metadata_features(stories, config, 0.0, 1)

    assert feats.shape == (1, 1)
    assert feats[0, 0] == pytest.approx(np.log1p(2.0) / np.log1p(15.0))
