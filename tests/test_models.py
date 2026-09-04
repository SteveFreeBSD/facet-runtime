from __future__ import annotations

import pytest

from facet_runtime import models


def test_every_backend_has_a_text_model() -> None:
    for backend in ("cpu", "gpu", "npu"):
        assignment = models.assignment(backend, "text")
        assert assignment.backend == backend
        assert assignment.model
        assert assignment.runtime in {"ollama", "fastflowlm"}
        assert assignment.rationale.strip()


def test_cpu_assignment_matches_installed_ollama_artifact() -> None:
    cpu = models.assignment("cpu", "text")

    assert cpu.model == "qwen3.5:2b"
    assert cpu.parameters == "2.3B"
    assert cpu.quantization == "Q8_0"
    assert cpu.disk_gib == 2.55


def test_npu_preference_records_observed_failure_without_guessing_its_cause() -> None:
    rationale = models.assignment("npu", "text").rationale

    assert "measured preferred NPU text worker" in rationale
    assert "returned an empty completion" in rationale
    assert "underlying cause was not captured" in rationale
    assert "rather than a confirmed cause" in rationale


def test_image_pipeline_backends_have_vision_models() -> None:
    for backend in ("gpu", "npu"):
        assert models.assignment(backend, "vision").model


def test_runtime_matches_the_device_it_serves() -> None:
    for assignment in models.ASSIGNMENTS:
        expected = "fastflowlm" if assignment.backend == "npu" else "ollama"
        assert assignment.runtime == expected


GTT_APERTURE_GIB = 14.8
USABLE_MEMORY_GIB = 29.0


def test_declared_footprint_fits_the_machine() -> None:
    for assignment in models.ASSIGNMENTS:
        assert assignment.disk_gib > 0
        if assignment.backend == "gpu":
            # Facet requires full GPU residency, so a GPU model that cannot fit
            # the aperture can only ever fail as a device mismatch.
            assert assignment.disk_gib < GTT_APERTURE_GIB
        # One model at a time, alongside the desktop and both runtimes.
        assert assignment.disk_gib < USABLE_MEMORY_GIB / 2


def test_unassigned_pairing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no vision model"):
        models.assignment("cpu", "vision")


def test_environment_override_replaces_only_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment = models.assignment("gpu", "text")
    monkeypatch.setenv(assignment.env_override, "some-other-model:1b")
    assert models.model_for("gpu") == "some-other-model:1b"
    assert models.assignment("gpu", "text").overridden() is True
    assert models.assignment("gpu", "text").backend == "gpu"


def test_report_lists_every_assignment() -> None:
    report = models.report()
    assert len(report) == len(models.ASSIGNMENTS)
    assert {(row["backend"], row["role"]) for row in report} == {
        (a.backend, a.role) for a in models.ASSIGNMENTS
    }
