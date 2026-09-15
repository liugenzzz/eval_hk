"""Conservative bilingual labels anchored by shared image identity.

Overrides use objects: [{source_domain, alias, canonical}] and defects:
[{source_domain, object_category, alias, canonical}]. Canonical values are
English names or ASCII ids. Original answer strings are never modified.
"""
import posixpath
import re
import unicodedata
from collections import defaultdict


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _canonical(value, object_label=False):
    value = normalize(value)
    if object_label:
        value = {"airplane": "aircraft", "aeroplane": "aircraft"}.get(value, value)
    if not value.isascii():
        return None
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or None


def extract_labels(record):
    answers = [x.get("value", "") for x in record.get("conversations", [])
               if x.get("from", "").lower() in ("gpt", "assistant")]
    if len(answers) != 1 or not isinstance(answers[0], str):
        raise ValueError("Expected exactly one assistant answer")
    match = re.match(
        r"^\s*(?:Category|类别)\s*[:：]\s*(.*?)\s*[;；]\s*"
        r"(?:Anomaly\s+type|异常类型)\s*[:：]\s*(.*?)\s*[;；]\s*"
        r"(?:Description|描述)\s*[:：]", answers[0], re.I | re.S)
    if not match or not all(x.strip() for x in match.groups()):
        raise ValueError("Missing structured Category / Anomaly type / Description prefix")
    return tuple(x.strip() for x in match.groups())


def image_identity(record):
    images = record.get("images", record.get("image"))
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str) or not images[0].strip():
        raise ValueError("Expected exactly one image path")
    path = posixpath.normpath(images[0].strip().replace("\\", "/"))
    marker = "defect-detect-images/"
    index = path.find(marker)
    key = path[index + len(marker):] if index >= 0 else path
    domain = key.split("/", 1)[0] if index >= 0 else posixpath.dirname(key)
    return key, domain


class LabelMapping:
    def __init__(self, en_records, zh_records, overrides=None):
        self.objects = defaultdict(set)
        self.defects = defaultdict(set)
        self.conflict_images = set()
        self.issues = []
        self.unresolved = []
        self.over_objects, self.over_defects = {}, {}
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict):
            raise ValueError("Mapping overrides must be an object")
        unknown = set(overrides) - {"objects", "defects"}
        if unknown:
            raise ValueError(f"Unknown mapping override keys: {sorted(unknown)}")
        for section, required in (("objects", ("source_domain", "alias", "canonical")),
                                  ("defects", ("source_domain", "object_category", "alias", "canonical"))):
            entries = overrides.get(section, [])
            if not isinstance(entries, list):
                raise ValueError(f"Mapping {section} must be a list")
            for entry in entries:
                if not isinstance(entry, dict) or any(
                    not isinstance(entry.get(field), str) or not entry[field].strip()
                    for field in required
                ):
                    raise ValueError(f"Mapping {section} entries require nonempty strings: {required}")
        for entry in overrides.get("objects", []):
            canonical = _canonical(entry["canonical"], True)
            if not canonical:
                raise ValueError("Override canonical must be an ASCII English name/id")
            key = (entry["source_domain"], normalize(entry["alias"]))
            if key in self.over_objects and self.over_objects[key] != canonical:
                raise ValueError(f"Conflicting object overrides for {key}")
            self.over_objects[key] = canonical
        for entry in overrides.get("defects", []):
            canonical = _canonical(entry["canonical"])
            obj = _canonical(entry["object_category"], True)
            if not canonical or not obj:
                raise ValueError("Override canonical and object_category must be ASCII")
            key = (entry["source_domain"], obj, normalize(entry["alias"]))
            if key in self.over_defects and self.over_defects[key] != canonical:
                raise ValueError(f"Conflicting defect overrides for {key}")
            self.over_defects[key] = canonical
        groups = {"en": defaultdict(set), "zh": defaultdict(set)}
        for language, records in (("en", en_records), ("zh", zh_records)):
            for record in records:
                try:
                    key, domain = image_identity(record)
                    obj, defect = extract_labels(record)
                    groups[language][(key, domain)].add((normalize(obj), normalize(defect)))
                except ValueError as error:
                    self.issues.append({"language": language, "reason": str(error)})
        for identity in groups["en"].keys() | groups["zh"].keys():
            en = groups["en"].get(identity, set())
            zh = groups["zh"].get(identity, set())
            canonical_en = set()
            for raw_obj, raw_defect in en:
                obj = self.over_objects.get((identity[1], raw_obj), _canonical(raw_obj, True))
                defect = self.over_defects.get((identity[1], obj, raw_defect), _canonical(raw_defect))
                canonical_en.add((obj, defect))
            if len(canonical_en) > 1 or len(zh) > 1:
                self.conflict_images.add(identity[0])
                continue
            if len(canonical_en) != 1 or len(zh) != 1:
                continue
            obj, defect = next(iter(canonical_en))
            if not obj or not defect:
                continue
            raw_obj, raw_defect = next(iter(zh))
            domain = identity[1]
            self.objects[(domain, raw_obj)].add(obj)
            self.defects[(domain, obj, raw_defect)].add(defect)
        for (key, domain), pairs in groups["zh"].items():
            for raw_obj, raw_defect in pairs:
                obj_key = (domain, raw_obj)
                objects = ({self.over_objects[obj_key]} if obj_key in self.over_objects
                           else self.objects.get(obj_key, set()))
                if len(objects) != 1:
                    self.unresolved.append({"image_key": key, "source_domain": domain,
                                            "raw_object_category": raw_obj,
                                            "raw_defect_type": raw_defect, "reason": "object_unresolved_or_ambiguous"})
                    continue
                defect_key = (domain, next(iter(objects)), raw_defect)
                if defect_key not in self.over_defects and len(self.defects.get(defect_key, set())) != 1:
                    self.unresolved.append({"image_key": key, "source_domain": domain,
                                            "raw_object_category": raw_obj,
                                            "raw_defect_type": raw_defect, "reason": "defect_unresolved_or_ambiguous"})

    def resolve(self, record, language):
        if language not in ("en", "zh"):
            raise ValueError("language must be en or zh")
        key, domain = image_identity(record)
        raw_obj, raw_defect = extract_labels(record)
        obj_key = (domain, normalize(raw_obj))
        sources = []
        ambiguous = key in self.conflict_images
        if obj_key in self.over_objects:
            obj = self.over_objects[obj_key]
            sources.append("manual")
        elif language == "en":
            obj = _canonical(raw_obj, True)
            sources.append("english")
        else:
            values = self.objects.get(obj_key, set())
            ambiguous |= len(values) > 1
            obj = next(iter(values)) if len(values) == 1 else None
            sources.append("shared_image")
        defect_key = (domain, obj, normalize(raw_defect))
        if defect_key in self.over_defects:
            defect = self.over_defects[defect_key]
            sources.append("manual")
        elif language == "en":
            defect = _canonical(raw_defect)
        else:
            values = self.defects.get(defect_key, set())
            ambiguous |= len(values) > 1
            defect = next(iter(values)) if len(values) == 1 else None
        # A conflicting image is unsafe even with a global alias override.
        status = "ambiguous" if ambiguous else "resolved" if obj and defect else "unresolved"
        return dict(image_key=key, source_domain=domain, object_category=obj,
                    defect_type=defect, raw_object_category=raw_obj,
                    raw_defect_type=raw_defect, mapping_source="+".join(sorted(set(sources))),
                    status=status)

    def audit(self):
        def entries(mapping, kind):
            return [{"key": list(key), "canonical_candidates": sorted(values),
                     "status": "resolved" if len(values) == 1 else "ambiguous", "kind": kind}
                    for key, values in sorted(mapping.items())]
        return {"objects": entries(self.objects, "object"),
                "defects": entries(self.defects, "defect"),
                "conflict_images": sorted(self.conflict_images), "invalid_records": self.issues,
                "unresolved": self.unresolved,
                "manual_overrides": {
                    "objects": [{"source_domain": key[0], "alias": key[1], "canonical": value}
                                for key, value in sorted(self.over_objects.items())],
                    "defects": [{"source_domain": key[0], "object_category": key[1],
                                 "alias": key[2], "canonical": value}
                                for key, value in sorted(self.over_defects.items())]}}


def build_label_mapping(en_records, zh_records, overrides=None):
    return LabelMapping(en_records, zh_records, overrides)
