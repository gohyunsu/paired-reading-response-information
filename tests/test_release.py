from paired_eval.report import verify_release


def test_release_assets_match_result_registry():
    report = verify_release()
    assert report["status"] == "pass"
    assert report["readme_figures"] == 4
    assert report["pdf_files"] == 0
