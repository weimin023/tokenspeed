# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tokenspeed_kernel.platform as platform_module
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    detect_rocm_interconnect,
)


@pytest.fixture
def detect_amd():
    def detect(arch):
        props = SimpleNamespace(
            gcnArchName=arch,
            name="AMD Radeon AI PRO R9700",
            total_memory=32 * 1024**3,
            multi_processor_count=32,
            max_shared_memory_per_block=64 * 1024,
        )
        with (
            patch("torch.cuda.get_device_properties", return_value=props),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device_count", return_value=1),
            patch.object(
                platform_module, "_get_rocm_runtime_features", return_value=frozenset()
            ),
        ):
            return platform_module._detect_rocm_platform()

    return detect


@pytest.mark.parametrize("arch", ["gfx1201", "gfx1201:sramecc-:xnack-"])
def test_gfx1201_detection(detect_amd, arch):
    platform = detect_amd(arch)
    assert platform.is_amd
    assert platform.arch_version == ArchVersion(12, 0)
    assert platform.generation_name == "RDNA4"
    assert platform.sm_features == frozenset()
    assert platform.max_shared_memory_per_sm == 64 * 1024
    assert platform.interconnect.topology == "single_gpu"
    assert not platform.is_cdna4
    assert not platform.is_cdna5
    assert not platform.is_cdna4_plus
    assert not platform.is_cdna5_plus


def test_gfx1201_uses_portable_capabilities(detect_amd):
    platform = detect_amd("gfx1201")
    assert CapabilityRequirement(vendors=frozenset({"amd"})).satisfied_by(platform)
    for feature in (
        "tensor_core:f16",
        "tensor_core:f8",
        "tensor_core:f4",
        "memory:async_copy",
    ):
        assert not CapabilityRequirement(
            required_features=frozenset({feature})
        ).satisfied_by(platform)
    for arch in (ArchVersion(9, 5), ArchVersion(12, 5)):
        assert not CapabilityRequirement(
            min_arch_version=arch, max_arch_version=arch
        ).satisfied_by(platform)


@pytest.mark.parametrize(
    "arch,version,generation,cdna5",
    [
        ("gfx950", ArchVersion(9, 5), "CDNA4", False),
        ("gfx1250", ArchVersion(12, 5), "CDNA5", True),
    ],
)
def test_cdna_detection_is_preserved(detect_amd, arch, version, generation, cdna5):
    platform = detect_amd(arch)
    assert platform.arch_version == version
    assert platform.generation_name == generation
    assert platform.is_cdna4_plus
    assert platform.is_cdna5_plus == cdna5
    assert platform.sm_features == frozenset(
        {"tensor_core:f16", "tensor_core:f8", "tensor_core:f4", "memory:async_copy"}
    )


def test_unknown_amd_architecture_is_rejected(detect_amd):
    with pytest.raises(
        RuntimeError, match="unsupported AMD GPU architecture 'gfx9999'"
    ):
        detect_amd("gfx9999")


def _pci_props(bus):
    return SimpleNamespace(pci_domain_id=0, pci_bus_id=bus, pci_device_id=0)


def _write_kfd_topology(nodes_dir, links):
    """Write CPU node 0 and GPU nodes 1..4; ``links`` maps (node, dir) to links.

    GPU node ``n`` sits on PCI bus ``n``; ``links[(n, "io_links")]`` is a list of
    ``(node_to, type)``.
    """
    cpu = nodes_dir / "0"
    cpu.mkdir(parents=True)
    (cpu / "properties").write_text("gfx_target_version 0\ndomain 0\nlocation_id 0\n")
    for node_id in range(1, 5):
        node_dir = nodes_dir / str(node_id)
        node_dir.mkdir()
        (node_dir / "properties").write_text(
            f"gfx_target_version 90500\ndomain 0\nlocation_id {node_id << 8}\n"
        )
        for link_dir_name in ("io_links", "p2p_links"):
            for index, (node_to, link_type) in enumerate(
                links.get((node_id, link_dir_name), ())
            ):
                link_dir = node_dir / link_dir_name / str(index)
                link_dir.mkdir(parents=True)
                (link_dir / "properties").write_text(
                    f"type {link_type}\nnode_to {node_to}\n"
                )


def _detect_interconnect(nodes_dir, visible_buses):
    with (
        patch("torch.cuda.device_count", return_value=len(visible_buses)),
        patch(
            "torch.cuda.get_device_properties",
            side_effect=lambda device: _pci_props(visible_buses[device]),
        ),
    ):
        return detect_rocm_interconnect(nodes_dir)


def _topology(interconnect):
    assert interconnect is not None
    return interconnect.topology


def test_rocm_interconnect_pcie(tmp_path):
    links = {(1, "io_links"): [(2, 2)], (2, "io_links"): [(1, 2)]}
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "pcie"


def test_rocm_interconnect_full_xgmi(tmp_path):
    links = {
        (1, "io_links"): [(2, 11), (3, 11)],
        (2, "io_links"): [(1, 11), (3, 11)],
        (3, "io_links"): [(1, 11), (2, 11)],
    }
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2, 3])) == "xgmi_full"


def test_rocm_interconnect_partial_xgmi(tmp_path):
    links = {(1, "io_links"): [(2, 11), (3, 2)]}
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2, 3])) == "xgmi_pairs"


def test_rocm_interconnect_counts_one_sided_xgmi_link(tmp_path):
    _write_kfd_topology(tmp_path, {(2, "io_links"): [(1, 11)]})

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "xgmi_full"


def test_rocm_interconnect_reads_p2p_links(tmp_path):
    _write_kfd_topology(tmp_path, {(1, "p2p_links"): [(2, 11)]})

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "xgmi_full"


def test_rocm_interconnect_p2p_link_does_not_hide_io_xgmi_link(tmp_path):
    links = {(1, "io_links"): [(2, 11)], (1, "p2p_links"): [(2, 2)]}
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "xgmi_full"


def test_rocm_interconnect_follows_visible_devices(tmp_path):
    # Only GPUs 1 and 2 are XGMI connected; GPUs 3 and 4 are not.
    links = {(1, "io_links"): [(2, 11)], (3, "io_links"): [(4, 2)]}
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "xgmi_full"
    assert _topology(_detect_interconnect(tmp_path, [3, 4])) == "pcie"
    # The visible order does not have to follow the KFD node order.
    assert _topology(_detect_interconnect(tmp_path, [4, 2])) == "pcie"


def test_rocm_interconnect_ignores_cpu_links(tmp_path):
    links = {(0, "io_links"): [(1, 11), (2, 11)], (1, "io_links"): [(0, 11)]}
    _write_kfd_topology(tmp_path, links)

    assert _topology(_detect_interconnect(tmp_path, [1, 2])) == "pcie"


def test_rocm_interconnect_single_gpu_does_not_read_kfd(tmp_path):
    with patch("torch.cuda.device_count", return_value=1):
        interconnect = detect_rocm_interconnect(tmp_path / "missing")

    assert _topology(interconnect) == "single_gpu"


def test_rocm_interconnect_unknown_without_kfd_topology(tmp_path):
    assert _detect_interconnect(tmp_path / "missing", [1, 2]) is None


def test_rocm_interconnect_unknown_when_visible_gpu_is_not_in_kfd(tmp_path):
    _write_kfd_topology(tmp_path, {})

    assert _detect_interconnect(tmp_path, [1, 9]) is None


def test_rocm_interconnect_unknown_when_link_file_is_unreadable(tmp_path):
    _write_kfd_topology(tmp_path, {(1, "io_links"): [(2, 11)]})
    (tmp_path / "1" / "io_links" / "0" / "properties").unlink()

    assert _detect_interconnect(tmp_path, [1, 2]) is None


def test_rocm_interconnect_unknown_when_link_file_is_malformed(tmp_path):
    _write_kfd_topology(tmp_path, {(1, "io_links"): [(2, 11)]})
    (tmp_path / "1" / "io_links" / "0" / "properties").write_text("node_to 2\n")

    assert _detect_interconnect(tmp_path, [1, 2]) is None
