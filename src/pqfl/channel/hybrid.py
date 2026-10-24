class HybridChannel:
    """Pending X25519 + ML-KEM-768 authenticated session establishment.

    TODO: transcript binding, identity authentication, standard combiner,
    directional traffic keys, replay protection, rotation and key erasure.
    This stub must never silently provide a classical-only channel.
    """

    def establish(self, *args, **kwargs):
        raise NotImplementedError("Hybrid channel specification pending")

    def rotate_keys(self):
        raise NotImplementedError("Rotation specification pending")
