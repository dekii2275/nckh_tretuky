import torch

from src.models.bilstm_attention import PaperBiLSTMAttentionClassifier


def test_paper_model_shapes_and_parameter_count():
    model = PaperBiLSTMAttentionClassifier(
        input_size=50,
        hidden_size=128,
        num_layers=2,
        num_classes=10,
        attention_heads=4,
        dropout=0.3,
        lstm_internal_dropout=0.0,
        attention_dropout=0.0,
    )
    inputs = torch.randn(2, 60, 50)
    lstm_output, _ = model.bilstm(inputs)
    attention_output, _ = model.attention(lstm_output, lstm_output, lstm_output)
    fused = model.layer_norm(lstm_output + attention_output)
    pooled = fused.mean(dim=1)
    projected = model.activation(model.projection(pooled))
    logits = model(inputs)

    assert lstm_output.shape == (2, 60, 256)
    assert attention_output.shape == (2, 60, 256)
    assert fused.shape == (2, 60, 256)
    assert pooled.shape == (2, 256)
    assert projected.shape == (2, 64)
    assert logits.shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 860362


def test_kinematic_v4_and_v5_models():
    from src.models.bilstm_attention import KinematicBiLSTMAttentionClassifier

    # Test v4 (110 dims)
    model_v4 = KinematicBiLSTMAttentionClassifier(raw_feature_dim=110)
    inputs = torch.randn(2, 60, 50, requires_grad=True)
    out_v4 = model_v4(inputs)
    assert out_v4.shape == (2, 10)
    assert model_v4.extract_kinematic_features(inputs).shape == (2, 60, 110)
    out_v4.sum().backward()
    assert torch.isfinite(inputs.grad).all()

    # Test v5 (125 dims)
    inputs.grad.zero_()
    model_v5_125 = KinematicBiLSTMAttentionClassifier(raw_feature_dim=125)
    out_v5_125 = model_v5_125(inputs)
    assert out_v5_125.shape == (2, 10)
    assert model_v5_125.extract_kinematic_features(inputs).shape == (2, 60, 125)
    out_v5_125.sum().backward()
    assert torch.isfinite(inputs.grad).all()

    # Test v5 (167 dims)
    inputs.grad.zero_()
    model_v5 = KinematicBiLSTMAttentionClassifier(raw_feature_dim=167)
    out_v5 = model_v5(inputs)
    assert out_v5.shape == (2, 10)
    assert model_v5.extract_kinematic_features(inputs).shape == (2, 60, 167)
    out_v5.sum().backward()
    assert torch.isfinite(inputs.grad).all()

