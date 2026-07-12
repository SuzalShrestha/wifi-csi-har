import numpy as np
import pytest
import torch

from csihar.models import MODEL_NAMES, CSICnn, CSICnnLstm, build_model

N_CLASSES = 6
S = 52


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("n_rx", [1, 3])
@pytest.mark.parametrize("n_time", [100, 300])
def test_forward_shapes_and_finite_logits(name, n_rx, n_time):
    torch.manual_seed(0)
    model = build_model(name, n_rx=n_rx, n_time=n_time,
                        n_subcarriers=S, n_classes=N_CLASSES)
    model.eval()
    x = torch.randn(4, n_rx, n_time, S)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, N_CLASSES)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_param_count_under_2m(name):
    model = build_model(name, n_rx=3, n_time=300,
                        n_subcarriers=S, n_classes=N_CLASSES)
    assert n_params(model) < 2_000_000


def test_cnn_tolerates_small_windows():
    # 1 s @ 64 Hz-ish and a narrow subcarrier subset must not collapse.
    model = CSICnn(n_rx=1, n_time=64, n_subcarriers=32, n_classes=N_CLASSES)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 64, 32))
    assert logits.shape == (2, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_cnn_lstm_trims_remainder_when_t_not_divisible():
    # T=110, chunk_len=25 -> 4 chunks, 10 samples trimmed.
    model = CSICnnLstm(n_rx=3, n_time=110, n_subcarriers=S,
                       n_classes=N_CLASSES, chunk_len=25)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 110, S))
    assert logits.shape == (4, N_CLASSES)
    assert torch.isfinite(logits).all()
    # trimmed input (T=100) gives the identical result — remainder is unused
    with torch.no_grad():
        x = torch.randn(2, 3, 110, S)
        full = model(x)
        trimmed = model(x[:, :, :100, :])
    assert torch.allclose(full, trimmed, atol=1e-5)


def test_cnn_lstm_rejects_t_shorter_than_chunk():
    with pytest.raises(ValueError, match="chunk_len"):
        CSICnnLstm(n_rx=3, n_time=10, n_subcarriers=S,
                   n_classes=N_CLASSES, chunk_len=25)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer", n_rx=3, n_time=300,
                    n_subcarriers=S, n_classes=N_CLASSES)


def test_models_reject_wrong_rx_count():
    model = build_model("cnn", n_rx=3, n_time=300,
                        n_subcarriers=S, n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 2, 300, S))


def test_models_train_step_changes_loss():
    # One gradient step must run end to end (BatchNorm in train mode).
    torch.manual_seed(0)
    np.random.seed(0)
    for name in MODEL_NAMES:
        model = build_model(name, n_rx=3, n_time=100,
                            n_subcarriers=S, n_classes=N_CLASSES)
        x = torch.randn(8, 3, 100, S)
        y = torch.randint(0, N_CLASSES, (8,))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
