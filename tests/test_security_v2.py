"""Regression checks for the conditional Jev-style HTTP decision model."""

import tempfile
import unittest
import json
import threading
import hashlib
from unittest.mock import patch
from urllib.request import Request, urlopen
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

# A tiny BERT exercises the same logits, losses and offline bundle path as the full model.
from security_jev_v2.model import SecurityJevBert, joint_loss, load_bundle, save_bundle
from security_jev_v2.predict import predict_loaded
from security_jev_v2.web import SecurityEngine, make_server
from security_jev_v2.evaluate import evaluate_bundle


class SecurityJevV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        vocab = root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nGET\n/\nHTTP\n1\n.\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab=str(vocab), do_lower_case=False)
        config = BertConfig(vocab_size=10, hidden_size=24, num_hidden_layers=1,
                            num_attention_heads=4, intermediate_size=48)
        self.model = SecurityJevBert(BertModel(config, add_pooling_layer=False)).eval()

    def tearDown(self):
        self.temp.cleanup()

    def test_two_heads_make_normalized_binary_and_conditional_choice(self):
        ids = torch.tensor([[2, 5, 6, 3], [2, 5, 7, 3]])
        output = self.model(ids, torch.ones_like(ids))
        self.assertEqual(output["block"].shape, (2,))
        self.assertEqual(output["types"].shape, (2, 4))
        p = torch.sigmoid(output["block"])
        self.assertTrue(torch.allclose(p + (1 - p), torch.ones_like(p)))
        self.assertTrue(torch.allclose(output["types"].softmax(-1).sum(-1), torch.ones(2)))

    def test_category_loss_ignores_benign_and_updates_encoder(self):
        ids = torch.tensor([[2, 5, 6, 3], [2, 5, 7, 3]])
        self.model.train()
        output = self.model(ids, torch.ones_like(ids))
        output["types"].retain_grad()
        losses = joint_loss(output, torch.tensor([0., 1.]), torch.tensor([-1, 2]))
        losses["total"].backward()
        self.assertTrue(torch.all(output["types"].grad[0] == 0))
        self.assertGreater(output["types"].grad[1].abs().sum().item(), 0)
        self.assertGreater(self.model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)

    def test_bundle_is_offline_and_v1_is_rejected(self):
        root = Path(self.temp.name)
        bundle = root / "bundle"
        save_bundle(self.model, self.tokenizer, bundle,
                    {"model_version": "bert-jev-security-v2", "max_length": 32,
                     "data_sha256": {"train": "a", "val": "b", "calib": "c", "test": "d"}},
                    {"block_temperature": 1.0, "block_bias": 0.0, "type_temperature": 1.0,
                     "threshold": 0.5, "status": "offline_experiment_only"})
        loaded, tokenizer, manifest, policy = load_bundle(bundle, "cpu")
        before = predict_loaded(self.model, self.tokenizer,
                                {"model_version": "bert-jev-security-v2", "max_length": 32},
                                policy, "GET / HTTP/1.1", "cpu")
        after = predict_loaded(loaded, tokenizer, manifest, policy, "GET / HTTP/1.1", "cpu")
        self.assertEqual(before["noul"]["probabilities"], after["noul"]["probabilities"])
        self.assertEqual(before["choice"]["probabilities"], after["choice"]["probabilities"])
        self.assertAlmostEqual(sum(after["choice"]["probabilities"].values()), 1.0, delta=1e-6)
        manifest["format_version"] = 1
        (bundle / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_bundle(bundle, "cpu")

    def test_v2_http_api_returns_same_scores_as_local_prediction(self):
        bundle = Path(self.temp.name) / "bundle"
        policy = {"block_temperature": 1.0, "block_bias": 0.0, "type_temperature": 1.0,
                  "threshold": 0.5, "status": "offline_experiment_only"}
        manifest = {"model_version": "test-v2", "max_length": 32, "model_source": "tiny-test"}
        save_bundle(self.model, self.tokenizer, bundle, manifest, policy)
        engine = SecurityEngine(bundle, "cpu")
        server = make_server("127.0.0.1", 0, engine)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        state = "GET / HTTP/1.1\nHost: example.test"
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/api/v2/predict",
                              data=json.dumps({"state": state}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request) as response:
                result = json.load(response)
            local = predict_loaded(engine.model, engine.tokenizer, engine.manifest,
                                   engine.policy, state, "cpu")
            self.assertEqual(result["noul"], local["noul"])
            self.assertEqual(result["choice"], local["choice"])
            self.assertIn("elapsed_ms", result)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_test_split_report_is_cached_after_first_evaluation(self):
        root = Path(self.temp.name)
        data = root / "data"
        data.mkdir()
        rows = [
            {"id": "normal", "state": "GET / HTTP/1.1", "block": 0,
             "primary_type": None, "site": "a.test", "source": "wcp", "origin": "source_legitimate"},
            {"id": "attack", "state": "GET /?q=bad HTTP/1.1", "block": 1,
             "primary_type": "XSS", "site": "b.test", "source": "wcp", "origin": "synthetic_payload"},
        ]
        test_file = data / "test.jsonl"
        test_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
        bundle = root / "bundle"
        save_bundle(self.model, self.tokenizer, bundle,
                    {"model_version": "test-v2", "model_source": "tiny-test", "max_length": 32,
                     "data_sha256": {"test": hashlib.sha256(test_file.read_bytes()).hexdigest()}},
                    {"block_temperature": 1.0, "block_bias": 0.0,
                     "type_temperature": 1.0, "threshold": .5,
                     "status": "offline_experiment_only"})
        first = evaluate_bundle(bundle, data, batch_size=2)
        with patch("security_jev_v2.evaluate.load_bundle", side_effect=AssertionError("rescored")):
            second = evaluate_bundle(bundle, data, batch_size=2)
        self.assertEqual(first, second)
        self.assertFalse(first["offline_gate_passed"])


if __name__ == "__main__":
    unittest.main()
