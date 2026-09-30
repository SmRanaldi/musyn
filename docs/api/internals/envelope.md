# Envelope internals (`musyn.envelope`)

The four stages behind {func}`musyn.extract_envelope`, in pipeline order.
See {doc}`../../user_guide/envelope` for the narrative version.

## 1. Pre-whitening

```{eval-rst}
.. automodule:: musyn.envelope.prewhiten
   :members:
```

## 2. Nu-order detection and window initialization

```{eval-rst}
.. automodule:: musyn.envelope.detection
   :members:
```

## 3. Adaptive loop and backend dispatch

```{eval-rst}
.. automodule:: musyn.envelope.adaptive
   :members:
```

## 4. High-level API

`extract_envelope` itself (the multi-channel `joblib` wrapper around the
three stages above) is documented at {func}`musyn.extract_envelope` in
{doc}`../public`.
