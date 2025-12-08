"""
Explainable AI (XAI) Module
===========================
SHAP and LIME implementations for model interpretability.
"""

from .shap_explain import (
    SHAPExplainer,
    LIMEExplainer,
    XAIReport,
    explain_prediction,
    SimpleSHAPFallback,
    SHAP_AVAILABLE,
    LIME_AVAILABLE
)

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'XAIReport',
    'explain_prediction',
    'SimpleSHAPFallback',
    'SHAP_AVAILABLE',
    'LIME_AVAILABLE'
]
