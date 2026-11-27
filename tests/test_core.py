import json
import math
import tempfile
import unittest
from pathlib import Path

from cryptonome.report import (
    script_safe_json,
    to_jsonable,
    write_html,
    write_json,
)


class ReportTests(unittest.TestCase):

    def test_nan_becomes_null(self):
        result = to_jsonable({
            "valid": 1.5,
            "invalid": float("nan"),
            "infinite": float("inf"),
        })

        self.assertEqual(result["valid"], 1.5)
        self.assertIsNone(result["invalid"])
        self.assertIsNone(result["infinite"])

    def test_outputs_and_hostile_input(self):
        payload = {
            "nodes": [
                {
                    "id": "w:W1",
                    "type": "wallet",
                    "label": "</script><script>alert('x')</script>",
                }
            ],
            "edges": [],
            "truncated": False,
        }

        safe = script_safe_json(payload)

        self.assertNotIn("</script>", safe.lower())
        self.assertIn("\\u003c", safe)

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)

            json_path = write_json(
                payload, folder / "report.json"
            )
            self.assertTrue(json_path.exists())

            saved = json.loads(
                json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(saved["nodes"]), 1)

            html_path = write_html(
                payload, folder / "report.html"
            )
            self.assertTrue(html_path.exists())

            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Cryptonome Investigation Report", html)
            self.assertNotIn(
                "</script><script>alert('x')",
                html.lower(),
            )


if __name__ == "__main__":
    unittest.main()
