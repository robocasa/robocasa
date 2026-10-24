from pathlib import PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

import pytest

from robocasa.scripts import download_datasets


@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath])
@pytest.mark.parametrize("custom_base", [False, True])
def test_box_link_lookup_uses_portable_archive_keys(
    monkeypatch, path_class, custom_base
):
    base = path_class("C:/datasets" if path_class is PureWindowsPath else "/datasets")
    package = base / "repo" / "robocasa"
    key = next(iter(download_datasets.BOX_LINKS_DS))
    destination = (
        (base if custom_base else package.parent / "datasets")
        / "v1.0"
        / key.removesuffix(".tar")
    )
    download = MagicMock()
    archive = MagicMock()
    open_archive = MagicMock(return_value=archive)
    remove = MagicMock()
    monkeypatch.setattr(download_datasets, "Path", path_class)
    monkeypatch.setattr(download_datasets.robocasa, "__path__", [str(package)])
    monkeypatch.setattr(
        download_datasets, "DATASET_BASE_PATH", str(base) if custom_base else None
    )
    monkeypatch.setattr(download_datasets.os, "makedirs", MagicMock())
    monkeypatch.setattr(download_datasets.os, "remove", remove)
    monkeypatch.setattr(download_datasets, "download_url", download)
    monkeypatch.setattr(download_datasets.tarfile, "open", open_archive)

    download_datasets.download_and_extract_from_box(str(destination))

    download.assert_called_once_with(
        url=download_datasets._get_direct_download_url(
            download_datasets.BOX_LINKS_DS[key]
        ),
        download_dir=str(destination.parent),
        fname=destination.name + ".tar",
        check_overwrite=False,
    )
    open_archive.assert_called_once()
    archive.__enter__.return_value.extractall.assert_called_once_with(
        path=destination.parent
    )
    remove.assert_called_once()
