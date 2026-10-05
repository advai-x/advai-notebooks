"""Compatibility inference functions; the public model facade owns inference."""


def predict_requests(bundle, requests, device, min_confidence=None, batch_size=8):
    return bundle.model.to(device).predict(requests, batch_size, min_confidence)


def predict_state(bundle, state, tasks, task, device, min_confidence=None):
    return bundle.model.to(device).predict_state(state, tasks, task, min_confidence)
