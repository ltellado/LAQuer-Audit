"""
Environment Bootstrapping and Utility Module.

Provides the `LAQuerAudit` class with static methods to:
1. Detect runtime environment (local VS Code / terminal vs. Google Colab).
2. Resolve repository root paths dynamically across platforms.
3. Automatically clone and pin the baseline LAQuer repository if missing.
4. Retrieve OpenAI API credentials from environment variables or Colab
   Secrets (`userdata`).
5. Select the best available PyTorch compute device (CUDA, MPS, CPU).
"""

import os
import subprocess
import sys
import torch


class LAQuerAudit:
    """
    Utility class for environment bootstrapping, path resolution,
    credential discovery, and compute device management.
    """

    GITHUB_REPO_URL = "https://github.com/ltellado/LAQuer-Audit.git"
    COLAB_REPO_DIR = "/content/LAQuer-Audit"

    @staticmethod
    def is_colab():
        """
        Check if current runtime is a Google Colab notebook instance.

        Output:
            bool: True if executing inside Google Colab, False otherwise.
        """
        return "google.colab" in sys.modules

    @staticmethod
    def get_repo_root():
        """
        Resolve the absolute path to the repository root directory.

        Output:
            str: Absolute path to the repository root.
        """
        if LAQuerAudit.is_colab():
            if os.path.exists(LAQuerAudit.COLAB_REPO_DIR):
                return LAQuerAudit.COLAB_REPO_DIR
            return "/content"

        return os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )

    @staticmethod
    def get_openai_api_key(prompt_if_missing=False):
        """
        Retrieve OpenAI API key from environment variables or Colab Secrets.

        Params:
            prompt_if_missing (bool): Whether to prompt user if key missing.

        Output:
            str or None: Retrieved API key string, or None if unavailable.
        """
        key = os.environ.get("OPENAI_API_KEY")
        if key and key.strip():
            return key.strip()

        if LAQuerAudit.is_colab():
            try:
                from google.colab import userdata # type: ignore

                key = userdata.get("OPENAI_API_KEY")
                if key and key.strip():
                    os.environ["OPENAI_API_KEY"] = key.strip()
                    return key.strip()
            except Exception:
                pass

        if prompt_if_missing:
            import getpass

            key = getpass.getpass("Enter your OPENAI_API_KEY: ").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
                return key

        return None

    @staticmethod
    def bootstrap_environment():
        """
        Ensure all directories, baseline submodules, and paths are ready.

        In Colab:
            - Clones the LAQuer-Audit repo if not yet cloned.
            - Runs setup.sh to pin eranhirs/LAQuer at commit a4affb2.
            - Adds repo root and src/ to sys.path and sets working directory.
            - Pre-loads OPENAI_API_KEY from Colab Secrets if present.

        In Local environments:
            - Verifies that the LAQuer baseline submodule is present.
            - Ensures repo root and src/ are in sys.path.

        Output:
            str: Resolved absolute path to the repository root directory.

        Raises:
            FileNotFoundError: If the pinned LAQuer directory is missing
                in a local environment where setup.sh was not yet run.
        """
        if LAQuerAudit.is_colab():
            repo_dir = LAQuerAudit.COLAB_REPO_DIR

            if not os.path.exists(repo_dir):
                print(
                    "Bootstrapping: Cloning "
                    + LAQuerAudit.GITHUB_REPO_URL
                    + " into "
                    + repo_dir
                    + "..."
                )
                subprocess.run(
                    [
                        "git",
                        "clone",
                        LAQuerAudit.GITHUB_REPO_URL,
                        repo_dir,
                    ],
                    check=True,
                )

            laquer_dir = os.path.join(repo_dir, "LAQuer")
            if not os.path.exists(laquer_dir):
                print("Bootstrapping: Running setup.sh in " + repo_dir + "...")
                subprocess.run(
                    ["bash", "setup.sh"], cwd=repo_dir, check=True
                )

            src_dir = os.path.join(repo_dir, "src")
            for p in (repo_dir, src_dir):
                if p not in sys.path:
                    sys.path.insert(0, p)
            os.chdir(repo_dir)

            LAQuerAudit.get_openai_api_key()
            return repo_dir

        repo_dir = LAQuerAudit.get_repo_root()
        laquer_dir = os.path.join(repo_dir, "LAQuer")

        if not os.path.exists(laquer_dir):
            raise FileNotFoundError(
                "Missing LAQuer submodule at: "
                + laquer_dir
                + ". Please run ./setup.sh from repository root."
            )

        src_dir = os.path.join(repo_dir, "src")
        for p in (repo_dir, src_dir):
            if p not in sys.path:
                sys.path.insert(0, p)

        return repo_dir

    @staticmethod
    def resolve_device():
        """
        Select best available PyTorch device across CUDA, MPS, and CPU.

        Output:
            torch.device: Configured torch device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")


if __name__ == "__main__":
    print("=== [Unit Test] LAQuerAudit ===")

    is_cloud = LAQuerAudit.is_colab()
    print("Is Google Colab:  " + str(is_cloud))

    root = LAQuerAudit.bootstrap_environment()
    print("Repository Root:  " + root)

    device = LAQuerAudit.resolve_device()
    print("PyTorch Device:   " + str(device))

    has_key = bool(LAQuerAudit.get_openai_api_key())
    print("API Key Detected: " + str(has_key))

    assert os.path.exists(root), "Repository root must exist."
    print("\n✓ Unit test passed for LAQuerAudit!\n")