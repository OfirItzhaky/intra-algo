import torch
import numpy as np
from typing import Callable, List, Optional
from research_agent.logging_setup import get_logger

log = get_logger(__name__)

class CNNFeatureInsightHelper:
    def __init__(self, model, X: torch.Tensor, y: torch.Tensor, feature_names: List[str], device: Optional[torch.device] = None):
        """
        Initializes the insight helper for CNN explainability.

        Args:
            model: TensorFlow Keras model (no .eval required)
            X (torch.Tensor): Input tensor [batch, seq_len, features]
            y (torch.Tensor): Ground truth labels
            feature_names (List[str]): List of feature names
            device (torch.device, optional): Torch device for saliency operations
        """
        self.model = model
        self.X = X.requires_grad_()
        self.y = y
        self.feature_names = feature_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def compute_saliency(self) -> np.ndarray:
        """
        Computes gradients of the output w.r.t input features using TensorFlow.
        Returns:
            np.ndarray: Absolute gradient values [batch, seq_len, features]
        """
        import tensorflow as tf

        x_input = tf.convert_to_tensor(self.X.detach().cpu().numpy())
        x_input = tf.cast(x_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_input)
            output = self.model(x_input)
            loss = tf.reduce_sum(output)

        gradients = tape.gradient(loss, x_input)
        self.saliency = torch.tensor(tf.abs(gradients).numpy())
        return self.saliency.numpy()

    def rank_features_by_influence(self, top_n: int = 10) -> None:
        """
        Ranks features by average gradient importance.

        Args:
            top_n (int): Number of top features to show
        """
        saliency = self.compute_saliency()
        importance = saliency.mean(axis=(0, 1))
        ranked_indices = np.argsort(importance)[::-1][:top_n]
        log.info("\n📊 Top Influential Features:")
        for i in ranked_indices:
            log.info(f"{self.feature_names[i]}: {importance[i]:.4f}")

    def explain_unused_features(self, threshold: float = 0.01) -> None:
        """
        Prints features with very low gradient activity (likely unused).

        Args:
            threshold (float): Minimum average gradient threshold
        """
        saliency = self.compute_saliency()
        avg_importance = saliency.mean(axis=(0, 1))
        unused = [self.feature_names[i] for i, val in enumerate(avg_importance) if val < threshold]
        if unused:
            log.info("\n🟡 Possibly Unused Features:")
            for f in unused:
                log.info(f"- {f}")
        else:
            log.info("\n✅ All features have some contribution above threshold.")

    def temporal_importance_summary(self) -> None:
        """
        Prints which timesteps (in the sequence) receive the most attention by the CNN.
        """
        saliency = self.compute_saliency()
        time_importance = saliency.mean(axis=(0, 2))
        importance_by_time = time_importance.mean(axis=0)
        peak_time = np.argmax(importance_by_time)
        log.info("\n⏳ Temporal Importance Summary:")
        log.info(f"- Peak attention at timestep: {peak_time}")
        log.info(f"- Last 5 steps gradient share: {importance_by_time[-5:].sum() / importance_by_time.sum():.2%}")

    def analyze_false_positives(self, predictions: torch.Tensor, threshold: float = 0.5) -> None:
        """
        Analyzes which features dominate in false positive predictions.

        Args:
            predictions (torch.Tensor): Model predicted probabilities
            threshold (float): Classification threshold
        """
        saliency = self.compute_saliency()
        preds = (predictions > threshold).float()
        mask_fp = (preds == 1) & (self.y == 0)
        if mask_fp.sum() == 0:
            log.info("\n✅ No false positives found in the current batch.")
            return
        saliency_fp = saliency[mask_fp.cpu().numpy()]
        avg_fp_importance = saliency_fp.mean(axis=(0, 1))
        top_fp_features = np.argsort(avg_fp_importance)[::-1][:5]
        log.info("\n⚠️ Features contributing most to false positives:")
        for i in top_fp_features:
            log.info(f"- {self.feature_names[i]}: {avg_fp_importance[i]:.4f}")

    def test_feature_removal_impact(self, X_original: torch.Tensor, y_true: torch.Tensor, metric_fn: Callable,
                                    threshold: float = 0.5) -> None:
        """
        Tests the effect of removing each feature on model performance.

        Args:
            X_original (torch.Tensor): Input tensor [batch, seq_len, features]
            y_true (torch.Tensor): Ground truth binary labels
            metric_fn (Callable): Metric function (e.g., precision_score)
            threshold (float): Classification threshold
        """
        log.info("\n📉 Metric Impact from Removing Each Feature:")

        # ✅ Get model prediction for full feature set
        base_output = np.squeeze(self.model(X_original.cpu().numpy()).numpy())
        base_preds = (base_output > threshold).astype(int)
        base_score = metric_fn(y_true.cpu().numpy(), base_preds)
        log.info(f"Base Score: {base_score:.4f}\n")

        # ✅ Loop through features and zero each
        for i, name in enumerate(self.feature_names):
            X_mod = X_original.clone()
            X_mod[:, :, i] = 0

            mod_output = np.squeeze(self.model(X_mod.cpu().numpy()).numpy())
            mod_preds = (mod_output > threshold).astype(int)
            mod_score = metric_fn(y_true.cpu().numpy(), mod_preds)
            delta = mod_score - base_score

            log.info(f"{name}: {mod_score:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})")

    def summarize_feature_gradient_impact(self) -> None:
        """
        Summarizes overall feature importance based on average saliency.
        """
        saliency = self.compute_saliency()
        avg_impact = saliency.mean(axis=(0, 1))
        sorted_idx = np.argsort(avg_impact)[::-1]
        log.info("\n🔬 CNN Gradient Impact by Feature:")
        for idx in sorted_idx:
            log.info(f"{self.feature_names[idx]}: {avg_impact[idx]:.4f}")

    def generate_llm_summary(self) -> None:
        """
        Placeholder for LLM-based natural language summary generation.
        """
        log.info("\n🤖 [LLM Summary Placeholder] Future: summarize results with GPT/Gemini here.")