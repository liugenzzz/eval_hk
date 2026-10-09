import json
import tempfile
import unittest
from pathlib import Path

from eval_tool.aircraft_eval_set import convert_record, convert_split, domain_of, size_bucket_of


def row(**changes):
    sample = {
        "qa_id": "q1", "sample_id": "image-1", "image": "/images/one.jpg",
        "image_hw": [640, 640], "task": "grounding_single",
        "question": "Where is the crack?", "answer": '[{"bbox_2d":[100,100,200,200],"label":"crack"}]',
        "image_status": "anomalous", "image_defect_types": ["crack"],
        "dataset": "uts_aircraft_defect", "target_type": "crack", "coord_mode": "norm1000",
    }
    sample.update(changes)
    return sample


class AircraftEvalSetTests(unittest.TestCase):
    def test_grounding_preserves_content_and_adds_scoring_metadata(self):
        source = row()
        result = convert_record(source)
        self.assertEqual(result["id"], "q1")
        self.assertEqual(result["images"], [source["image"]])
        self.assertEqual(result["conversations"][0]["value"], "<image>" + source["question"])
        self.assertEqual(result["conversations"][1]["value"], source["answer"])
        self.assertNotIn("system", result)
        self.assertNotIn("meta", result)
        self.assertEqual(result["metadata"]["size_bucket"], "medium")
        self.assertEqual(result["metadata"]["source_image"], "image-1")

    def test_classification_requires_target(self):
        self.assertIsNone(convert_record(row(task="classification_open", target_type=None)))
        result = convert_record(row(task="classification_mc", answer="A", target_type="crack"))
        self.assertEqual(result["metadata"]["target_type"], "crack")

    def test_counterfactual_and_negative(self):
        counterfactual = convert_record(row(task="grounding_counterfactual", target_type=None,
                                            absent_type="corrosion", answer="[]"))
        self.assertEqual(counterfactual["metadata"]["defect_type"], "corrosion")
        self.assertNotIn("size_bucket", counterfactual["metadata"])
        negative = convert_record(row(task="grounding_negative", target_type=None,
                                      image_status="normal", image_defect_types=[], answer="[]"))
        self.assertEqual(negative["metadata"]["defect_type"], "normal")

    def test_domain_filter_and_task_filter(self):
        transfer = row(dataset="visa", task="discrimination")
        self.assertIsNone(convert_record(transfer))
        self.assertEqual(convert_record(transfer, "all")["metadata"]["domain"], "transfer")
        self.assertIsNone(convert_record(row(task="description")))
        self.assertEqual(domain_of("synthetic_panel"), "synthetic")

    def test_box_validation_and_buckets(self):
        self.assertEqual(size_bucket_of('[{"bbox_2d":[0,0,50,50]}]'), "small")
        self.assertEqual(size_bucket_of('[{"bbox_2d":[0,0,500,500]}]'), "large")
        self.assertIsNone(size_bucket_of('[{"bbox_2d":[200,0,100,50]}]'))
        self.assertIsNone(convert_record(row(answer='not JSON')))

    def test_train_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "train.raw.jsonl"
            source.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only held-out"):
                convert_split(source, directory / "output.jsonl")


if __name__ == "__main__":
    unittest.main()
