"""Pure-numpy inference for the DeepCare Keras MLP (no TensorFlow or scikit-learn needed).

Pipeline for one or more people given as RAW feature values in cfg['feature_order']:
    raw -> clip BMI -> standardize numeric columns -> MLP forward pass -> Platt calibration
"""
import json
from pathlib import Path

import numpy as np


def load_bundle(models_dir):
    d = Path(models_dir)
    bundle = json.loads((d / 'app_bundle.json').read_text())
    weights = {k: v.astype('float64') for k, v in np.load(d / 'mlp_weights.npz').items()}
    cfg = json.loads((d / 'app_config.json').read_text())
    background = np.load(d / 'shap_background.npz')['data'].astype('float64')   # already in scaled model space
    return {'bundle': bundle, 'weights': weights, 'cfg': cfg, 'background': background}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def preprocess(raw, art):
    """RAW values, shape (n, n_features) in feature_order -> scaled model input."""
    cfg, sc = art['cfg'], art['bundle']['scaler']
    order = cfg['feature_order']
    x = np.array(raw, dtype='float64', copy=True)
    if x.ndim == 1:
        x = x[None, :]
    j = order.index('BMI')
    lo, hi = cfg['bmi_clip']
    x[:, j] = np.clip(x[:, j], lo, hi)
    for name, mean, scale in zip(sc['features'], sc['mean'], sc['scale']):
        k = order.index(name)
        x[:, k] = (x[:, k] - mean) / scale
    return x


def raw_score(x_scaled, art):
    """MLP forward pass: probability before calibration."""
    W = art['weights']
    h = np.asarray(x_scaled, dtype='float64')
    for layer in art['bundle']['layers']:
        t = layer['type']
        if t == 'dense':
            h = h @ W[layer['w']] + W[layer['b']]
            if layer['activation'] == 'sigmoid':
                h = _sigmoid(h)
            elif layer['activation'] == 'relu':
                h = np.maximum(h, 0.0)
            elif layer['activation'] != 'linear':
                raise ValueError(f"unsupported activation {layer['activation']}")
        elif t == 'bn':
            h = (h - W[layer['mm']]) / np.sqrt(W[layer['mv']] + layer['eps']) * W[layer['g']] + W[layer['be']]
        elif t == 'relu':
            h = np.maximum(h, 0.0)
        else:
            raise ValueError(f'unsupported layer type {t}')
    return h.ravel()


def calibrate(p, art):
    """Platt scaling fitted on the validation set: sigmoid(coef * logit(p) + intercept)."""
    p = np.clip(np.asarray(p, dtype='float64'), 1e-6, 1 - 1e-6)
    pl = art['bundle']['platt']
    return _sigmoid(pl['coef'] * np.log(p / (1 - p)) + pl['intercept'])


def predict_risk(x_scaled, art):
    return calibrate(raw_score(x_scaled, art), art)
