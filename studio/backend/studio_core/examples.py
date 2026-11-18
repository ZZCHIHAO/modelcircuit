"""Fresh, editable S1 synthetic experiment documents."""


def example_graph():
    """Return a new two-branch classifier with one shared encoder instance."""
    return {
        "schema_version": 1, "id": "shared_classifier", "name": "共享双路分类器", "revision": 1,
        "inputs": [
            {"id": "a", "name": "输入 A", "shape": [None, 4], "dtype": "float32"},
            {"id": "b", "name": "输入 B", "shape": [None, 4], "dtype": "float32"},
        ],
        "instances": [
            {"id": "encoder", "kind": "chip", "definition_id": "encoder_chip", "name": "共享编码器", "params": {"in_features": 4, "hidden_features": 8}, "frozen": False},
            {"id": "fusion", "kind": "Add", "name": "相加融合", "params": {}},
            {"id": "head", "kind": "Linear", "name": "分类头", "params": {"in_features": 8, "out_features": 2}, "frozen": False},
        ],
        "calls": [
            {"id": "encode_a", "instance_id": "encoder", "inputs": {"x": "input:a"}, "position": {"x": 250, "y": 90}},
            {"id": "encode_b", "instance_id": "encoder", "inputs": {"x": "input:b"}, "position": {"x": 250, "y": 300}},
            {"id": "fuse", "instance_id": "fusion", "inputs": {"a": "call:encode_a", "b": "call:encode_b"}, "position": {"x": 540, "y": 190}},
            {"id": "classify", "instance_id": "head", "inputs": {"x": "call:fuse"}, "position": {"x": 800, "y": 190}},
        ],
        "outputs": [{"id": "logits", "source": "call:classify"}, {"id": "features", "source": "call:fuse"}],
        "definitions": [{
            "id": "encoder_chip", "name": "编码芯片", "version": "0.1.0",
            "public_parameters": [
                {"id": "in_features", "name": "输入维度", "default": 4},
                {"id": "hidden_features", "name": "特征维度", "default": 8},
            ],
            "bindings": {"in_features": ["dense.in_features"], "hidden_features": ["dense.out_features"]},
            "graph": {
                "inputs": [{"id": "x", "name": "特征输入", "shape": [None, 4], "dtype": "float32"}],
                "instances": [
                    {"id": "dense", "kind": "Linear", "name": "特征投影", "params": {"in_features": 4, "out_features": 8}},
                    {"id": "activation", "kind": "ReLU", "name": "激活", "params": {}},
                ],
                "calls": [
                    {"id": "project", "instance_id": "dense", "inputs": {"x": "input:x"}, "position": {"x": 220, "y": 180}},
                    {"id": "activate", "instance_id": "activation", "inputs": {"x": "call:project"}, "position": {"x": 520, "y": 180}},
                ],
                "outputs": [{"id": "out", "source": "call:activate"}],
            },
        }],
    }
