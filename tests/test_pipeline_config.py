from eval_tool.artifacts import ArtifactLayout


def test_artifact_layout_derives_every_pipeline_path(tmp_path):
    layout = ArtifactLayout(work_dir=tmp_path / "work", out_dir=tmp_path / "report")

    assert layout.model_dir("base") == tmp_path / "work" / "base"
    assert layout.prediction("base", "aero_vqa") == (
        tmp_path / "work" / "base" / "base_aero_vqa.xlsx"
    )
    assert layout.partial_dir("base") == tmp_path / "work" / "base" / "_partial"
    assert layout.manifest("base", "aero_vqa") == (
        tmp_path / "work" / "base" / "base_aero_vqa.infer.json"
    )
    assert layout.rubric_out("v4") == tmp_path / "report_v4"
