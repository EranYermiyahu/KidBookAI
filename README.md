# KidBookAI

KidBookAI crafts personalized, high-resolution storybooks featuring AI-generated
artwork of each child alongside tailored narratives derived from form inputs.

## AI Photo Generation Block

This block integrates with [Replicate](https://replicate.com/) to transform a
child's reference photo into scene-specific, storybook-quality illustrations.

### Key Concepts

- `src/ai_generation/prompting.py` formats the structured prompt that keeps the
  child's identity intact while adapting the environment, wardrobe, and mood to
  each scene.
- `src/ai_generation/replicate_service.py` wraps the Replicate client,
  validating configuration and dispatching generation requests.

### Getting Started

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure environment variables (or provide arguments in code):
   - `REPLICATE_API_TOKEN`
   - `REPLICATE_MODEL` (format: `owner/model:version`)
3. Generate an image (example usage):
   ```python
   from src.ai_generation import ReplicateImageGenerator

   generator = ReplicateImageGenerator()
   result = generator.generate_image(
       kid_name="Aiden",
       scene_description="Aiden soars over a starlit city wearing a glowing superhero cape.",
       input_image="/path/to/aiden-reference.jpg",
       guidance_scale=3.5,
   )

   for url in result:
       print(url)
   ```

### Continuity & Identity Controls

- Use `src/pipeline/continuity.py`'s `IllustrationContinuityConfig` to keep the
  protagonist and recurring characters consistent across pages.
- The config feeds deterministic seeds, identity notes, and reference history
  into the pipeline. Reference history defaults to the `reference_image_history`
  parameter used by InstantID-style models—override `reference_history_parameter`
  if your Replicate model expects a different input name.
- Pass the config when creating the orchestrator:

  ```python
  from src import IllustrationContinuityConfig, KidBookAIOrchestrator

  continuity = IllustrationContinuityConfig(
      identity_notes=(
          "- Light brown skin with warm undertones\n"
          "- Curly dark hair in a shoulder-length bob\n"
          "- Hazel eyes with faint freckles"
      ),
      supporting_character_notes={
          "Milo the puppy": "small golden retriever pup, teal bandana",
          "Grandma Neta": "silver hair in a bun, rounded glasses, lavender cardigan",
      },
      static_continuity_notes=[
          "Keep the child's teal explorer jacket unless the scene explicitly changes wardrobe.",
      ],
      reference_history_size=3,
      promote_latest_reference=True,
  )

  orchestrator = KidBookAIOrchestrator(continuity_config=continuity)
  package = orchestrator.run_from_profile_mapping(
      profile_data,
      desired_pages=14,
      reference_image="example_images/laura_girl.jpg",
  )
  ```

  The orchestrator now threads the identity snapshot into every prompt, reuses
  prior illustration URLs as extra references, and keeps deterministic seeds
  stable unless you override them in `image_kwargs`.

> **Note:** The Replicate model chosen should support identity-preserving
> image-to-image editing (e.g., InstantID-based workflows).

## Story Generation Block

This block consumes the structured kid profile (converted from the Hebrew
Google Form responses) and produces a complete story draft via LiteLLM-compatible chat models (OpenAI by default).

- `src/story_generation/profile.py` normalizes the raw JSON/YAML responses into
  a `KidProfile` dataclass.
- `src/story_generation/prompting.py` creates the system/user prompt pair that
  drives the LLM.
- `src/story_generation/story_service.py` routes prompts through LiteLLM to request the
  finished story Markdown from whichever provider/model you configure.

Example usage:

```python
from pathlib import Path
import yaml

from src.story_generation import KidProfile, StoryOutlineGenerator

profile_data = yaml.safe_load(Path("kid_profile.yaml").read_text())
profile = KidProfile.from_mapping(profile_data)

generator = StoryOutlineGenerator()
story_markdown = generator.generate_story(profile)

print(story_markdown)
```

Ensure the relevant provider API key is available (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, Azure credentials, etc.) so LiteLLM can authenticate. Optionally set one of the model env vars (`KIDBOOKAI_STORY_MODEL`, `OPENAI_STORY_MODEL`, or `LITELLM_MODEL`) to override the default.

## Story Pagination & Scene Bridging

After generating the full story:

- `src/story_generation/page_splitter.py` uses `StoryPageSplitter` to break the
  Markdown into 12-18 illustration-ready pages while keeping the narrative
  intact.
- `src/story_generation/scene_builder.py` converts each page into a concise
  `SceneDescription` fit for the Replicate image prompt.

Example flow:

```python
pages = StoryPageSplitter().split_story(
    story_markdown,
    profile=profile,
    desired_pages=14,
)

scene_generator = SceneDescriptionGenerator()
scenes = [scene_generator.page_to_scene(page, profile=profile) for page in pages]

for scene in scenes:
    print(scene.page_number, scene.scene_description)
```

Optional environment overrides:
- `KIDBOOKAI_PAGE_MODEL`, `OPENAI_PAGE_SPLITTER_MODEL`, or `LITELLM_PAGE_MODEL`
- `KIDBOOKAI_SCENE_MODEL`, `OPENAI_SCENE_MODEL`, or `LITELLM_SCENE_MODEL`

## End-to-End Pipeline

- `src/pipeline/pipeline.py` exposes `KidBookAIOrchestrator`, which links profile parsing, story generation, pagination, scene conversion, and image creation. Pass `story_model`, `page_model`, or `scene_model` (and matching API keys if needed) to override models per block without touching downstream code.

Example MVP flow:

```python
from pathlib import Path
import yaml

from src import KidBookAIOrchestrator

profile_data = yaml.safe_load(Path("kid_profile.yaml").read_text())

orchestrator = KidBookAIOrchestrator()
package = orchestrator.run_from_profile_mapping(
    profile_data,
    desired_pages=14,
    reference_image="example_images/laura_girl.jpg",
    image_kwargs={"guidance_scale": 3.0},
)

Path("kidbook_package.yaml").write_text(package.to_yaml(), encoding="utf-8")
```

The resulting YAML contains the full story, page-by-page text, scene descriptions, and the image outputs returned by Replicate. Configure `REPLICATE_*` along with the appropriate LiteLLM provider keys/model env vars before running.

To run from the command line:

```bash
python scripts/run_full_pipeline.py \
  --profile kid_profile.yaml \
  --reference-image example_images/laura_girl.jpg \
  --output kidbook_package.yaml
```

## PDF Layout Block

Once the YAML package is ready, you can render a printable storybook PDF:

- `src/pdf_generation/builder.py` contains `StorybookPDFBuilder`, which arranges a
  cover, centred story spreads, and illustration spreads in an alternating rhythm.
- Text pages include only the narrative (`story_text`) styled with colourful read-aloud
  backgrounds; illustration pages focus solely on the artwork.
- `scripts/render_story_pdf.py` is a CLI wrapper that downloads the illustration URLs,
  applies the layout, and writes a high-resolution PDF that is square by default and
  ready for print-on-demand workflows.

Example usage:

```bash
python scripts/render_story_pdf.py \
  --package kidbook_package.yaml \
  --output kidbook_story.pdf \
  --page-size square
```
