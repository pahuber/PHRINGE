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

    def has_tensor(self, obj) -> bool:
        """
        Recursively check if 'obj' (which may be nested lists/tuples)
        contains at least one torch.Tensor.
        """
        if isinstance(obj, torch.Tensor):
            return True
        elif isinstance(obj, (list, tuple)):
            return any(self.has_tensor(x) for x in obj)
        return False

    def is_signature_equal(self, sig1, sig2) -> bool:
        """
        If neither sig1 nor sig2 contains a torch.Tensor,
        compare them with normal '=='.
        Otherwise, do a minimal custom comparison.
        """
        # Quick short-circuit: if type is different, not equal
        if type(sig1) != type(sig2):
            return False

        # If neither has Tensor => normal '=='
        if not self.has_tensor(sig1) and not self.has_tensor(sig2):
            return sig1 == sig2

        # If either has a Tensor, do a minimal custom approach
        return self.eq_for_signature(sig1, sig2)

    def eq_for_signature(self, old, new) -> bool:
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
            return all(self.eq_for_signature(o, n) for o, n in zip(old, new))
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
        attribute_name = f'cache_{attribute_name}'

        # Derive the signature attribute name from attribute_name
        sig_attribute_name = f"_sig_{attribute_name}"

        # Retrieve any existing signature & cached value
        old_sig = getattr(self, sig_attribute_name, None)
        cached_value = self._cache.get(attribute_name, None)
        # getattr(self, attribute_name, None)

        # Build the new signature from the current dependencies
        new_sig = tuple(required_attributes)
        new_sig = new_sig.__str__()

        # If there's an old signature that matches new_sig, return the existing cache
        if old_sig is not None and self.is_signature_equal(old_sig, new_sig):
            return cached_value

        # Otherwise, something changed => recompute
        new_value = compute_func()

        # Store updated cache and signature
        # setattr(self, attribute_name, new_value)
        self._cache[attribute_name] = new_value
        setattr(self, sig_attribute_name, new_sig)
        return new_value
