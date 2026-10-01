import torch

from src.models.bilstm_attention import BiLSTMAttentionClassifier


def test_bilstm_attention_output_shape():
    model = BiLSTMAttentionClassifier(50, 8, 1, 4, attention_heads=4, dropout=0.0)
    inputs = torch.randn(2, 5, 25, 2)
    lengths = torch.tensor([5, 3])
    logits = model(inputs, lengths=lengths)
    assert logits.shape == (2, 4)


def test_attention_ignores_padding():
    torch.manual_seed(0)
    model = BiLSTMAttentionClassifier(50, 8, 1, 3, attention_heads=4, dropout=0.0)
    model.eval()
    inputs = torch.randn(2, 5, 50)
    lengths = torch.tensor([5, 2])
    baseline = model(inputs, lengths=lengths)
    changed = inputs.clone()
    changed[1, 2:] = 10000.0
    perturbed = model(changed, lengths=lengths)
    assert torch.allclose(baseline[1], perturbed[1], atol=1e-5, rtol=1e-5)


def test_invalid_attention_dimension_is_rejected():
    try:
        BiLSTMAttentionClassifier(50, 7, 1, 3, attention_heads=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected invalid attention dimension to fail")


def test_kinematic_bilstm_attention_forward_and_backward():
    from src.models.bilstm_attention import KinematicBiLSTMAttentionClassifier
    model = KinematicBiLSTMAttentionClassifier(input_size=50, hidden_size=64, num_layers=1, num_classes=10)
    inputs = torch.randn(4, 30, 25, 2)
    logits = model(inputs)
    assert logits.shape == (4, 10)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


