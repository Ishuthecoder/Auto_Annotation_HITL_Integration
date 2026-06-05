from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("hitl.imgserver")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


class ImageServer:
    """Multi-directory image lookup with 4-strategy filename resolution."""

    def __init__(self, dirs: List[Path]):
        self.dirs = [d for d in dirs if d.exists()]
        missing = [d for d in dirs if not d.exists()]
        for d in missing:
            log.warning("Image directory not found: %s", d)

        self.stem_index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        count = 0
        for d in self.dirs:
            for p in d.rglob("*"):
                if p.suffix.lower() in IMG_EXTENSIONS:
                    key = p.stem.lower()
                    if key not in self.stem_index:
                        self.stem_index[key] = p
                    count += 1
        log.info(
            "Stem index: %d images across %d dirs", count, len(self.dirs)
        )

    def find(self, file_name: str) -> Optional[Path]:
        p = Path(file_name)
        stem = p.stem.lower()

        # Strategy 1: exact stem
        if stem in self.stem_index:
            return self.stem_index[stem]

        # Strategy 2: strip trailing underscores
        stripped = stem.rstrip("_")
        if stripped != stem and stripped in self.stem_index:
            return self.stem_index[stripped]

        # Strategy 3: direct path / basename / extension variants
        for d in self.dirs:
            for candidate in [
                d / file_name,
                d / p.name,
                *[d / (p.stem + ext) for ext in IMG_EXTENSIONS],
            ]:
                if candidate.exists():
                    return candidate

        # Strategy 4: fuzzy prefix match (last resort)
        for key, path in self.stem_index.items():
            if key.startswith(stem) or stem.startswith(key):
                log.debug("Fuzzy match '%s' → '%s'", stem, key)
                return path

        log.warning("Image not found: '%s'", file_name)
        return None

    def serve(self, file_name: str) -> Tuple[Optional[bytes], str]:
        path = self.find(file_name)
        if path is None:
            return None, ""
        ext = path.suffix.lower()
        mime = MIME_MAP.get(ext, "image/jpeg")
        try:
            return path.read_bytes(), mime
        except Exception as e:
            log.error("Failed to read %s: %s", path, e)
            return None, ""
