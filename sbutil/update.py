# updater_github_main.py
# GitHub main ブランチを参照してアドオンをセルフアップデートする簡易実装

from __future__ import annotations

import ast
import io
import os
import re
import shutil
import tempfile
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import bpy
from bpy.app.handlers import persistent


# -----------------------------
# 設定（Preferences から上書きする想定）
# -----------------------------

@dataclass
class GithubRepo:
    owner: str = "abev-crypto"
    repo: str = "SkybrushUtil"
    branch: str = "main"
    # リポジトリ内でアドオンが置かれている相対パス（repo 直下なら空でOK）
    # 例: "src/my_addon" の場合は "src/my_addon"
    addon_subdir: str = "sbutil"


# -----------------------------
# ユーティリティ
# -----------------------------

def _http_get_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BlenderAddonUpdater/1.0",
            "Accept": "text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data.decode("utf-8", errors="replace")


def _http_get_bytes(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BlenderAddonUpdater/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_bl_info_version_from_init_py(py_text: str) -> Optional[Tuple[int, int, int]]:
    """
    __init__.py の bl_info = {... "version": (x,y,z) ... } を安全に読む（ast で解析）
    """
    tree = ast.parse(py_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "bl_info":
                    # bl_info の dict literal を期待
                    if isinstance(node.value, (ast.Dict,)):
                        d = ast.literal_eval(node.value)
                        v = d.get("version")
                        if (
                            isinstance(v, tuple)
                            and len(v) >= 2
                            and all(isinstance(i, int) for i in v[:3])
                        ):
                            if len(v) == 2:
                                return (v[0], v[1], 0)
                            return (v[0], v[1], v[2])
    return None


def _version_gt(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    return a > b


def _addon_root_dir_from_module(module_name: str) -> str:
    """
    bpy.context.preferences.addons[module].module のルートパスを引く
    """
    mod = __import__(module_name)
    # package の __file__ は __init__.py を指す
    p = os.path.dirname(os.path.abspath(mod.__file__))
    return p


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _copytree_overwrite(src: str, dst: str) -> None:
    """
    dst を消してから src を丸ごとコピー（最もトラブルが少ない）
    """
    tmp_old = dst + ".old"
    # 既存を退避
    if os.path.exists(dst):
        _safe_rmtree(tmp_old)
        os.rename(dst, tmp_old)

    shutil.copytree(src, dst)
    _safe_rmtree(tmp_old)
# -----------------------------
# メイン処理
# -----------------------------

def get_local_version(module_name: str) -> Optional[Tuple[int, int, int]]:
    mod = __import__(module_name)
    bl_info = getattr(mod, "bl_info", None)
    if isinstance(bl_info, dict):
        v = bl_info.get("version")
        if isinstance(v, tuple) and len(v) >= 2:
            if len(v) == 2:
                return (int(v[0]), int(v[1]), 0)
            return (int(v[0]), int(v[1]), int(v[2]))
    return None


def get_remote_version(repo: GithubRepo) -> Optional[Tuple[int, int, int]]:
    # raw URL: https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/__init__.py
    sub = repo.addon_subdir.strip("/")

    init_path = "__init__.py" if not sub else f"{sub}/__init__.py"
    url = f"https://raw.githubusercontent.com/{repo.owner}/{repo.repo}/{repo.branch}/{init_path}"
    text = _http_get_text(url)
    return _parse_bl_info_version_from_init_py(text)


def download_main_zip(repo: GithubRepo) -> bytes:
    # https://github.com/{owner}/{repo}/archive/refs/heads/main.zip
    url = f"https://github.com/{repo.owner}/{repo.repo}/archive/refs/heads/{repo.branch}.zip"
    return _http_get_bytes(url)


def install_from_zip_bytes(zip_bytes: bytes, repo: GithubRepo, target_addon_dir: str) -> None:
    """
    zip を展開して、repo.addon_subdir にあるアドオンディレクトリ内容を target_addon_dir に上書きする
    """
    sub = repo.addon_subdir.strip("/")

    with tempfile.TemporaryDirectory() as td:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        zf.extractall(td)

        # zip の最上位は "{repo}-{branch}/" になるのが通例
        # 例: myrepo-main/
        top_dirs = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
        if not top_dirs:
            raise RuntimeError("ZIPの展開先にトップディレクトリが見つかりませんでした。")
        top = os.path.join(td, top_dirs[0])

        src_addon_dir = top if not sub else os.path.join(top, sub)
        if not os.path.isdir(src_addon_dir):
            raise RuntimeError(f"ZIP内に addon_subdir が見つかりません: {src_addon_dir}")

        # src_addon_dir の中身がアドオンとして妥当か最低限チェック
        if not os.path.exists(os.path.join(src_addon_dir, "__init__.py")):
            raise RuntimeError("ZIP内の対象フォルダに __init__.py がありません。addon_subdir 設定を確認してください。")

        # 置換（丸ごと）
        _copytree_overwrite(src_addon_dir, target_addon_dir)


# -----------------------------
# Blender Operator / UI
# -----------------------------

_AUTO_CHECK_DONE = False


def _addon_key() -> str:
    return __package__.split(".")[0]


def _get_prefs() -> Optional[bpy.types.AddonPreferences]:
    return bpy.context.preferences.addons[_addon_key()].preferences
def _check_update_for_prefs(prefs: bpy.types.AddonPreferences) -> Tuple[bool, str]:
    repo = GithubRepo(
        owner=prefs.gh_owner,
        repo=prefs.gh_repo,
        branch=prefs.gh_branch,
        addon_subdir=prefs.gh_addon_subdir,
    )

    local_v = get_local_version(_addon_key())
    remote_v = get_remote_version(repo)
    prefs.last_local_version = str(local_v) if local_v else "None"
    prefs.last_remote_version = str(remote_v) if remote_v else "None"

    if not local_v or not remote_v:
        prefs.update_available = False
        return False, "Missing version info"

    prefs.update_available = _version_gt(remote_v, local_v)
    return True, "Update available" if prefs.update_available else "Up to date"


def _notify_update_available(message: str) -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        print(message)
        return

    def _draw(self, _context):
        self.layout.label(text=message)

    wm.popup_menu(_draw, title="SkyBrushUtil", icon='INFO')
@persistent
def _auto_check_on_load(_scene) -> None:
    global _AUTO_CHECK_DONE
    if _AUTO_CHECK_DONE:
        return
    prefs = _get_prefs()
    if prefs is None or not getattr(prefs, "auto_check", False):
        _AUTO_CHECK_DONE = True
        return
    ok, _msg = _check_update_for_prefs(prefs)
    if ok and getattr(prefs, "update_available", False):
        _notify_update_available("SkyBrushUtil update available. Open Preferences to update.")
    _AUTO_CHECK_DONE = True


def register():
    if _auto_check_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_auto_check_on_load)


def unregister():
    if _auto_check_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_auto_check_on_load)
