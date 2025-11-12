"""
Integration with Replicate for high-fidelity storybook image generation.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import replicate

from .prompting import StorybookPrompt, build_storybook_prompt


class ReplicateImageGenerator:
    """
    Convenience wrapper around the Replicate client for storybook image generation.

    Parameters
    ----------
    api_token:
        Replicate API token. Falls back to ``REPLICATE_API_TOKEN`` environment variable.
    model_identifier:
        Fully-qualified model string in the ``owner/model:version`` format. Falls back to
        ``REPLICATE_MODEL`` environment variable. A value must be provided from one of the
        two sources.
    client:
        Optional pre-configured :class:`replicate.Client`. Mainly useful for testing.
    """

    def __init__(
        self,
        *,
        api_token: str | None = None,
        model_identifier: str | None = None,
        client: replicate.Client | None = None,
    ) -> None:
        self._api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self._api_token and not client:
            raise ValueError(
                "Replicate API token is required. Set REPLICATE_API_TOKEN or pass api_token."
            )

        self._model_identifier = model_identifier or os.getenv("REPLICATE_MODEL")
        if not self._model_identifier:
            raise ValueError(
                "Replicate model identifier is required. "
                "Set REPLICATE_MODEL or pass model_identifier in the form 'owner/model:version'."
            )

        self._client = client or replicate.Client(api_token=self._api_token)

    @property
    def model_identifier(self) -> str:
        """Return the model identifier currently used."""
        return self._model_identifier

    def generate_image(
        self,
        *,
        kid_name: str,
        scene_description: str,
        input_image: str | Path | BinaryIO,
        camera_shot: str | None = None,
        negative_prompt_override: str | None = None,
        **model_kwargs: Any,
    ) -> Iterable[Any]:
        """
        Generate an image from Replicate using the configured model.

        Parameters
        ----------
        kid_name:
            Name of the child featuring in the scene.
        scene_description:
            Narrative description of the scene to render.
        input_image:
            Either a local file path, URL, or binary file-like object pointing to the kid's reference image.
        camera_shot:
            Optional override for the camera framing instructions in the prompt.
        negative_prompt_override:
            Optional custom negative prompt to replace the default guardrails.
        **model_kwargs:
            Additional keyword arguments forwarded directly to the Replicate model invocation.

        Returns
        -------
        Iterable[Any]
            Raw output produced by Replicate. Most image models return an iterable of URLs.
        """
        prompt: StorybookPrompt = build_storybook_prompt(
            kid_name=kid_name,
            scene_description=scene_description,
            camera_shot=camera_shot,
        )

        with ExitStack() as stack:
            image_input = _prepare_image_input(input_image, stack=stack)

            replicate_input: dict[str, Any] = {
                "prompt": prompt.positive,
                "negative_prompt": (
                    negative_prompt_override
                    if negative_prompt_override is not None
                    else prompt.negative
                ),
                "image": image_input,
            }

            # Allow the caller to tweak model-specific knobs (e.g., guidance_scale, seed).
            replicate_input.update(model_kwargs)

            return self._client.run(
                self._model_identifier,
                input=replicate_input,
            )


def _prepare_image_input(
    input_image: str | Path | BinaryIO,
    *,
    stack: ExitStack,
) -> str | BinaryIO:
    """
    Normalize the image input so Replicate can consume it, keeping resources open via ExitStack.
    """
    if hasattr(input_image, "read"):
        # Assume file-like object, rely on caller to manage its lifecycle.
        return input_image  # type: ignore[return-value]

    if isinstance(input_image, Path):
        input_path = input_image.expanduser()
    else:
        input_candidate = str(input_image)
        input_path = Path(input_candidate).expanduser()
        if input_candidate.lower().startswith(("http://", "https://")):
            return input_candidate

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found at '{input_path}'.")

    file_handle = stack.enter_context(input_path.open("rb"))
    return file_handle


