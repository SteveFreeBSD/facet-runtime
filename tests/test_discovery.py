from facet_runtime.discovery import discover, discover_cpu


def test_cpu_is_always_available() -> None:
    cpu = discover_cpu()
    assert cpu["available"] is True
    assert cpu["logical_cpus"]


def test_report_has_expected_backends() -> None:
    assert set(discover()["backends"]) == {"cpu", "gpu", "npu"}
