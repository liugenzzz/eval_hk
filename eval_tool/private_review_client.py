"""Project adapter for the bundled private-model router (no secret persistence)."""
from argparse import Namespace
from pathlib import Path
from . import _private_model_router as router


class PrivateReviewClient:
    def __init__(self, provider=None, config=None):
        self.provider = provider
        self.config = config if config is not None else router.load_config()

    @property
    def cache_settings(self):
        # Canonical configuration contains environment VARIABLE NAMES, never values.
        return {"provider": self.provider, "pool": self.config["vlm_pool"]}

    def review(self, prompt, image_paths=()):
        args = Namespace(prompt=prompt, prompt_file=None, image=[str(Path(p)) for p in image_paths],
                         task_type="industrial_review", provider=self.provider, system=None,
                         dry_run=False, raw=False)
        try:
            result = router.run_chat(args, self.config)
        except Exception as exc:
            # HTTP errors may echo request headers; never persist their text.
            raise RuntimeError("Private review request failed (" + type(exc).__name__ + ")") from None
        if not isinstance(result.get("content"), str) or not result["content"].strip():
            raise RuntimeError("Private review returned non-text or empty content")
        return result
