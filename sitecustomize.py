"""
Site customization to fix package import order in Triton server.

The Triton DALI backend includes its own wheel directory at the front of sys.path,
which contains an incomplete 'packaging' module. This causes import errors when
other packages (like hydra-core) try to use packaging.

This script ensures system site-packages are prioritized over backend wheels.
"""
import sys


def _reorder_path_for_triton():
    """Reorder sys.path to prioritize system packages over backend wheels."""
    site_packages = []
    backend_wheels = []
    other_paths = []

    for path in sys.path:
        if not path:
            continue
        # Identify system site-packages and dist-packages
        if "site-packages" in path or "dist-packages" in path:
            site_packages.append(path)
        # Identify backend wheel directories (like DALI)
        elif "/backends/" in path and "/wheel/" in path:
            backend_wheels.append(path)
        else:
            other_paths.append(path)

    # Rebuild sys.path: other paths first, then site-packages, then backend wheels last
    # This ensures complete packages from site-packages are found before incomplete ones
    sys.path[:] = other_paths + site_packages + backend_wheels


_reorder_path_for_triton()
