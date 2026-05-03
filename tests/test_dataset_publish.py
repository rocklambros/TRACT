"""Tests for tract.dataset.publish — HuggingFace dataset upload."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tract.dataset.publish import publish_dataset


class TestDryRun:
    @patch("tract.dataset.publish.subprocess")
    def test_dry_run_skips_upload(
        self, mock_subprocess: MagicMock, tmp_path: Path,
    ) -> None:
        publish_dataset("user/repo", tmp_path, dry_run=True)
        mock_subprocess.check_output.assert_not_called()


class TestSkipUpload:
    @patch("tract.dataset.publish.subprocess")
    def test_skip_upload_skips_upload(
        self, mock_subprocess: MagicMock, tmp_path: Path,
    ) -> None:
        publish_dataset("user/repo", tmp_path, skip_upload=True)
        mock_subprocess.check_output.assert_not_called()


class TestCreateRepo:
    @patch("huggingface_hub.HfApi")
    @patch(
        "tract.dataset.publish.subprocess.check_output",
        return_value="fake-token\n",
    )
    def test_create_repo_called_with_dataset_type(
        self,
        mock_pass: MagicMock,
        mock_hfapi_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        publish_dataset("user/repo", tmp_path)

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/repo", repo_type="dataset", exist_ok=True,
        )


class TestUploadFolder:
    @patch("huggingface_hub.HfApi")
    @patch(
        "tract.dataset.publish.subprocess.check_output",
        return_value="fake-token\n",
    )
    def test_upload_folder_called_with_correct_args(
        self,
        mock_pass: MagicMock,
        mock_hfapi_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        publish_dataset("user/repo", tmp_path)

        mock_api.upload_folder.assert_called_once_with(
            folder_path=str(tmp_path),
            repo_id="user/repo",
            repo_type="dataset",
        )


class TestTokenCleanup:
    @patch("huggingface_hub.HfApi")
    @patch(
        "tract.dataset.publish.subprocess.check_output",
        return_value="fake-token\n",
    )
    def test_token_deleted_on_success(
        self,
        mock_pass: MagicMock,
        mock_hfapi_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api

        publish_dataset("user/repo", tmp_path)

        mock_hfapi_cls.assert_called_once_with(token="fake-token")

    @patch("huggingface_hub.HfApi")
    @patch(
        "tract.dataset.publish.subprocess.check_output",
        return_value="fake-token\n",
    )
    def test_token_deleted_on_upload_failure(
        self,
        mock_pass: MagicMock,
        mock_hfapi_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hfapi_cls.return_value = mock_api
        mock_api.upload_folder.side_effect = RuntimeError("upload failed")

        with pytest.raises(RuntimeError, match="upload failed"):
            publish_dataset("user/repo", tmp_path)

        mock_api.create_repo.assert_called_once()
