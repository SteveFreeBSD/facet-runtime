from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from facet_runtime.adapters.base import ImageAdapterOutput, ImageRuntimeMetadata
from facet_runtime.errors import BackendMismatchError
from facet_runtime.image_pipeline import inspect_image


@dataclass
class FakeImageAdapter:
    backend: str
    transcription: str
    uncertainties: tuple[str, ...] = ()
    verified: bool = True
    calls: list[str] = field(default_factory=list)

    def inspect_image(self, image_path: str) -> ImageAdapterOutput:
        self.calls.append(image_path)
        return ImageAdapterOutput(
            backend=self.backend,
            transcription=self.transcription,
            uncertainties=self.uncertainties,
            model="qwen3.5:4b",
            runtime=f"fake-{self.backend}",
            device=f"fake-{self.backend}-device",
            runtime_metadata=ImageRuntimeMetadata(
                protocol="fake",
                response_format="fake",
                strict_json_schema=False,
            ),
            accelerator_verified=self.verified,
        )


def _fixture(tmp_path) -> str:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"deterministic fixture")
    return str(image)


def test_pipeline_returns_two_verified_passes_and_agreement(tmp_path) -> None:
    image = _fixture(tmp_path)
    npu = FakeImageAdapter("npu", "line one\nx^2")
    gpu = FakeImageAdapter("gpu", "line one\nx^2")
    result = inspect_image(image, npu_adapter=npu, gpu_adapter=gpu)
    assert result.agreement is True
    assert result.disagreement == ()
    assert result.npu.accelerator_verified is True
    assert result.gpu.accelerator_verified is True
    assert npu.calls == gpu.calls == [result.image]
    assert len(result.image_sha256) == 64
    assert result.to_dict().keys() == {
        "image",
        "image_sha256",
        "npu",
        "gpu",
        "agreement",
        "disagreement",
        "total_elapsed_ms",
    }


def test_pipeline_reports_line_level_disagreement_and_uncertainty(tmp_path) -> None:
    image = _fixture(tmp_path)
    result = inspect_image(
        image,
        npu_adapter=FakeImageAdapter(
            "npu", "equation: sqrt(17)", ("radical style normalized",)
        ),
        gpu_adapter=FakeImageAdapter("gpu", "equation: √17"),
    )
    assert result.agreement is False
    assert result.disagreement[0].npu_line == 1
    assert result.disagreement[0].gpu_line == 1
    assert result.disagreement[0].npu == "equation: sqrt(17)"
    assert result.disagreement[0].gpu == "equation: √17"
    assert result.npu.uncertainties == ("radical style normalized",)


def test_pipeline_aligns_an_inserted_title_without_cascading(tmp_path) -> None:
    image = _fixture(tmp_path)
    result = inspect_image(
        image,
        npu_adapter=FakeImageAdapter("npu", "Project: Facet\nx^2"),
        gpu_adapter=FakeImageAdapter("gpu", "TITLE\nProject: Facet\nx^2"),
    )
    assert len(result.disagreement) == 1
    assert result.disagreement[0].npu_line is None
    assert result.disagreement[0].gpu_line == 1
    assert result.disagreement[0].gpu == "TITLE"


def test_pipeline_refuses_unverified_or_wrong_accelerator(tmp_path) -> None:
    image = _fixture(tmp_path)
    gpu = FakeImageAdapter("gpu", "unused")
    with pytest.raises(BackendMismatchError, match="reported gpu, expected npu"):
        inspect_image(
            image,
            npu_adapter=FakeImageAdapter("gpu", "wrong backend"),
            gpu_adapter=gpu,
        )
    assert gpu.calls == []

    with pytest.raises(BackendMismatchError, match="did not verify"):
        inspect_image(
            image,
            npu_adapter=FakeImageAdapter("npu", "unverified", verified=False),
            gpu_adapter=gpu,
        )
