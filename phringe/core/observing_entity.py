import torch

from phringe.core.base_entity import BaseEntity


class ObservingEntity(BaseEntity):
    """
    A base class that provides a method `_get_cached_value` for caching derived values
    using a signature built from a tuple of dependencies. This allows lazy caching if they dependent attributed do not
    change and provides a mechanism to automatically recalculate the derived (cached) attribute if they do.

    - We store the cached result in `cache_attr_name`, e.g. "_derived_cache".
    - We store the last signature in an automatically generated name,
      e.g. "_sig__derived_cache".

    If the new signature matches the old one, we return the cached value.
    Otherwise, we recalc via `compute_func`.
    """
    # TODO: add usage example
    _cache: dict = {}

    def __init__(self, **data):
        super().__init__(**data)

    def _compare_tensors(self, old, new) -> bool:
        """
        Minimal custom comparison for Tensors & nested structures.
        """
        if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
            if old.shape != new.shape or old.dtype != new.dtype:
                return False
            return bool(torch.allclose(old, new))
        elif isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
            if len(old) != len(new):
                return False
            return all(self._compare_tensors(o, n) for o, n in zip(old, new))
        else:
            return old == new

    def _get_cached_value(
            self,
            attribute_name: str,
            compute_func,
            required_attributes: tuple
    ):
        """
        :param compute_func: A zero-arg function that returns the newly computed derived value.
        :param required_attributes: A tuple of the current dependency values (attributes) that affect the result.

        The signature is stored in an attribute named "_sig_{attribute_name}".
        """
        # Derive the signature attribute name from attribute_name
        sig_attribute_name = f"_sig_{attribute_name}"

        # Retrieve any existing signature & cached value
        old_sig = getattr(self, sig_attribute_name, None)
        cached_value = self._cache.get(attribute_name, None)
        # getattr(self, attribute_name, None)

        # Build the new signature from the current dependencies
        new_sig = required_attributes

        # If there's an old signature that matches new_sig, return the existing cache
        if old_sig is not None and self._is_signature_equal(old_sig, new_sig):
            return cached_value

        # Otherwise, something changed => recompute
        new_value = compute_func()

        # Store updated cache and signature
        # setattr(self, attribute_name, new_value)
        self._cache[attribute_name] = new_value
        setattr(self, sig_attribute_name, new_sig)
        return new_value

    def _has_tensor(self, obj) -> bool:
        """
        Recursively check if 'obj' (which may be nested lists/tuples)
        contains at least one torch.Tensor.
        """
        if isinstance(obj, torch.Tensor):
            return True
        elif isinstance(obj, (list, tuple)):
            return any(self._has_tensor(x) for x in obj)
        return False

    def _is_signature_equal(self, sig1, sig2) -> bool:
        """
        If neither sig1 nor sig2 contains a torch.Tensor,
        compare them with normal '=='.
        Otherwise, do a minimal custom comparison.
        """
        # Quick short-circuit: if type is different, not equal
        if type(sig1) != type(sig2):
            return False

        # If neither has Tensor => normal '=='
        if not self._has_tensor(sig1) and not self._has_tensor(sig2):
            return sig1 == sig2

        # If either has a Tensor, do a minimal custom approach
        return self._compare_tensors(sig1, sig2)


def observing_property(*, observed_attributes: tuple = ()):
    """
    Defines a property that:
      1. Uses `_get_cached_value` internally for lazy, signature-based caching.
      2. `attribute_name` is the cache key (plus a signature key).
      3. `compute_func` is a function that computes the value (e.g. your _get_something method).
      4. `observed_attributes` is a tuple of dependencies for the signature.

    Usage:
        @observing_property(
            attribute_name="spectral_energy_distribution",
            compute_func=lambda self: self._get_spectral_energy_distribution(),
            observed_attributes=(
                lambda s: s._instrument._field_of_view,
                lambda s: s._instrument.wavelength_bin_centers,
                # etc...
            )
        )
        def _spectral_energy_distribution(self):
            pass

    NOTE:
      - We pass `observed_attributes` as callables (lambdas) so that
        they are evaluated at runtime (in the getter), not at class definition time.
      - The decorated function body can be empty or contain docstrings.
    """

    def decorator(method):
        def getter(self):
            # Evaluate each dependency callable to get the current values
            deps = tuple(fn(self) for fn in observed_attributes)

            # if None in deps:
            #     raise ValueError("One of the observed attributes is None")

            # Now call your signature-based caching method
            return self._get_cached_value(
                attribute_name=method.__qualname__,
                compute_func=lambda: method(self),
                required_attributes=deps
            )

        # Preserve docstring from the original method (if any)
        return property(getter, doc=method.__doc__)

    return decorator
