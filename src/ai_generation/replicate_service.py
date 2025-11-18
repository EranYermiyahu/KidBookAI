"""
Integration with Replicate for high-fidelity storybook image generation.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence

import replicate

from .prompting import StorybookPrompt, build_storybook_prompt


def _build_instant_id_input(
    *,
    prompt: StorybookPrompt,
    negative_prompt: str,
    image_input: str | BinaryIO,
) -> dict[str, Any]:
    return {
        "prompt": prompt.positive,
        "negative_prompt": negative_prompt,
        "image": image_input,
        "output_format": "png",
        "sdxl_weights": "protovision-xl-high-fidel",
        "guidance_scale": 5,
        # "num_inference_steps": 50,
        # "ip_adapter_scale": 1.1,
        # "controlnet_conditioning_scale": 1.1,
        # "output_quality": 95,
    }


def _build_flux_kontext_input(
    *,
    prompt: StorybookPrompt,
    negative_prompt: str,
    image_input: str | BinaryIO,
) -> dict[str, Any]:
    return {
        "prompt": prompt.positive,
        "negative_prompt": negative_prompt,
        "input_image": image_input,
        "output_format": "png",
        "safety_tolerance": 2,
        "prompt_upsampling": True,
        "aspect_ratio": "1:1",
    }


_MODEL_INPUT_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "zsxkib/instant-id": _build_instant_id_input,
    "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789": _build_instant_id_input,
    "black-forest-labs/flux-kontext-pro": _build_flux_kontext_input,
}


def _build_replicate_input_payload(
    *,
    model_identifier: str,
    prompt: StorybookPrompt,
    negative_prompt: str,
    image_input: str | BinaryIO,
) -> dict[str, Any]:
    normalized_identifier = model_identifier.strip().lower()
    builder = _MODEL_INPUT_BUILDERS.get(normalized_identifier)
    if builder is None and ":" in normalized_identifier:
        base_identifier = normalized_identifier.split(":", maxsplit=1)[0]
        builder = _MODEL_INPUT_BUILDERS.get(base_identifier)
    if builder is None:
        supported_models = ", ".join(sorted(set(_MODEL_INPUT_BUILDERS)))
        raise ValueError(
            "Model identifier "
            f"'{model_identifier}' is not configured with a default input payload. "
            f"Supported models: {supported_models}."
        )

    return builder(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_input=image_input,
    )


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
        favorite_theme: str | Sequence[str] | None = None,
        outfit_description: str | None = None,
        facial_expression: str | Sequence[str] | None = None,
        pose_description: str | Sequence[str] | None = None,
        input_image: str | Path | BinaryIO,
        camera_shot: str | None = None,
        negative_prompt_override: str | None = None,
        identity_traits: str | Sequence[str] | None = None,
        continuity_notes: Sequence[str] | None = None,
        supporting_cast_notes: Sequence[str] | Mapping[str, str] | None = None,
        reference_image_history: Sequence[str | Path | BinaryIO] | None = None,
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
        favorite_theme:
            Optional thematic guidance (e.g., "Superhero adventures") that should color
            the scene's overall style, palette, and wardrobe motifs.
        outfit_description:
            Wardrobe guidance inferred for this page. Clothing should follow this
            description rather than the reference photo.
        facial_expression:
            Optional prescribed facial expression cues tailoring the child's emotion
            to this moment in the story.
        pose_description:
            Optional pose or body-language guidance aligned with the scene action.
        input_image:
            Either a local file path, URL, or binary file-like object pointing to the kid's reference image.
        camera_shot:
            Optional override for the camera framing instructions in the prompt.
        negative_prompt_override:
            Optional custom negative prompt to replace the default guardrails.
        identity_traits:
            Structured facts about the child (e.g., age, wardrobe notes) that should be restated
            in the prompt to reinforce likeness preservation.
        continuity_notes:
            Carry-over cues from earlier pages to maintain continuity across the story.
        supporting_cast_notes:
            Optional list or mapping describing recurring supporting characters and their traits.
        reference_image_history:
            Optional iterable of prior illustration references (URLs or file paths) that should
            reinforce continuity when the underlying model supports multiple references.
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
            favorite_theme=favorite_theme,
            outfit_description=outfit_description,
            facial_expression=facial_expression,
            pose_description=pose_description,
            identity_traits=identity_traits,
            continuity_notes=continuity_notes,
            supporting_cast_notes=supporting_cast_notes,
        )

        with ExitStack() as stack:
            image_input = _prepare_image_input(input_image, stack=stack)

            negative_prompt = (
                negative_prompt_override
                if negative_prompt_override is not None
                else prompt.negative
            )

            replicate_input = _build_replicate_input_payload(
                model_identifier=self._model_identifier,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_input=image_input,
            )

            if reference_image_history:
                replicate_input["reference_image_history"] = [
                    _prepare_image_input(item, stack=stack)
                    for item in reference_image_history
                ]

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


