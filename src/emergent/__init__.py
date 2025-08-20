# Auto-register the first-principles RG beta (no-op if import fails)
try:
    from . import rg_fp  # side-effect: installs beta_exact into emergent.rg.beta_function
except Exception:
    pass
