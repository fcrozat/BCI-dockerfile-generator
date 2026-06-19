import pytest
from unittest.mock import patch, MagicMock
from bci_build.container_attributes import Arch
from bci_build.package.nvidia import NVIDIA_CONTAINERS, _get_driver_branch
from bci_build.repomdparser import RpmPackage

def test_nvidia_dkms_presence():
    """Verify that dkms is present in the third party package list for all driver branches >= 575."""
    for container in NVIDIA_CONTAINERS:
        branch = _get_driver_branch(container.version)
        dkms_pkgs = [p for p in container.third_party_package_list if p.name == "dkms"]
        if branch >= 575:
            assert len(dkms_pkgs) == 1, f"Expected dkms package for branch {branch} in container {container.name}"
        else:
            assert len(dkms_pkgs) == 0, f"Did not expect dkms package for branch {branch} in container {container.name}"


@patch("bci_build.package.nvidia.CUSTOM_END_TEMPLATE")
def test_nvidia_kmp_exclusion(mock_template):
    """Verify that SLES KMP packages are excluded from the closed driver builder layer."""
    # Find a 595 container
    container = next(c for c in NVIDIA_CONTAINERS if _get_driver_branch(c.version) >= 595)

    # Mock fetch_rpm_packages to return a predefined set of packages
    dummy_rpm = RpmPackage(
        name="nvidia-driver-G07",
        arch="x86_64",
        evr=("", "595.71.05", "1.1"),
        filename="nvidia-driver-G07-595.71.05-1.1.x86_64.rpm",
        url="http://dummy/nvidia-driver-G07-595.71.05-1.1.x86_64.rpm"
    )
    kmp_rpm = RpmPackage(
        name="nvidia-open-driver-G07-signed-cuda-kmp-default",
        arch="x86_64",
        evr=("", "595.71.05_k6.4.0", "1.1"),
        filename="nvidia-open-driver-G07-signed-cuda-kmp-default-595.71.05_k6.4.0-1.1.x86_64.rpm",
        url="http://dummy/nvidia-open-driver-G07-signed-cuda-kmp-default-595.71.05_k6.4.0-1.1.x86_64.rpm"
    )

    with patch.object(container, "fetch_rpm_packages", return_value=[dummy_rpm]) as mock_fetch, \
         patch("bci_build.package.nvidia._get_nvidia_kmp_rpms", return_value=[kmp_rpm]) as mock_kmp, \
         patch("bci_build.package.nvidia._get_kernel_ga_rpms", return_value=[]) as mock_kernel, \
         patch.object(container, "prepare_extra_files") as mock_prep_extra:

        # We also need to mock super(DevelopmentContainer, container).prepare_template()
        # so it doesn't try to build the whole Dockerfile which requires other side-effects.
        with patch("bci_build.package.DevelopmentContainer.prepare_template") as mock_super_prepare:
            container.prepare_template()

            # Ensure expected mocks were called
            mock_fetch.assert_called_once()
            mock_kmp.assert_called_once()
            mock_kernel.assert_called_once()

            # Get the args passed to render
            mock_template.render.assert_called_once()
            render_kwargs = mock_template.render.call_args[1]

            get_closed_packages_for_arch = render_kwargs["get_closed_packages_for_arch"]
            get_open_packages_for_arch = render_kwargs["get_open_packages_for_arch"]

            # Evaluate get_closed_packages_for_arch on x86_64
            closed_packages = get_closed_packages_for_arch(Arch.X86_64)
            closed_pkg_names = [p.name for p in closed_packages]

            # The closed driver list should contain nvidia-driver-G07 but NOT the KMP signed open driver
            assert "nvidia-driver-G07" in closed_pkg_names
            assert "nvidia-open-driver-G07-signed-cuda-kmp-default" not in closed_pkg_names

            # Evaluate get_open_packages_for_arch on x86_64
            open_packages = get_open_packages_for_arch(Arch.X86_64)
            open_pkg_names = [p.name for p in open_packages]

            # SLES KMP signed open driver should be in open driver packages, but closed driver should not
            assert "nvidia-open-driver-G07-signed-cuda-kmp-default" in open_pkg_names
            assert "nvidia-driver-G07" not in open_pkg_names


def test_get_nvidia_kmp_rpms():
    """Verify that _get_nvidia_kmp_rpms successfully returns RpmPackage objects without errors for SL16_0 and SP7."""
    from bci_build.package.nvidia import _get_nvidia_kmp_rpms
    from bci_build.os_version import OsVersion

    # Test SL16_0
    rpms_16 = _get_nvidia_kmp_rpms("595.71.05", OsVersion.SL16_0, "default", [Arch.X86_64])
    assert len(rpms_16) == 1
    assert rpms_16[0].name == "nvidia-open-driver-G07-signed-cuda-kmp-default"

    # Test SP7
    rpms_15 = _get_nvidia_kmp_rpms("595.71.05", OsVersion.SP7, "default", [Arch.X86_64])
    assert len(rpms_15) == 1
    assert rpms_15[0].name == "nvidia-open-driver-G07-signed-cuda-kmp-default"

