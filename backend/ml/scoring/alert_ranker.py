import pandas as pd


class WalletAlertRanker:
    """
    Ranks wallet alerts using risk and confidence scores.
    Higher priority alerts appear first.
    """

    def rank_alerts(self, alerts):
        required_columns = {
            "wallet",
            "risk_score",
            "confidence_score",
            "risk_level"
        }

        if not required_columns.issubset(alerts.columns):
            raise ValueError(
                f"Missing required columns: {required_columns}"
            )

        if alerts.empty:
            raise ValueError("Alert data cannot be empty.")

        if alerts["wallet"].isna().any():
            raise ValueError("Wallet identifiers cannot be missing.")

        if alerts["wallet"].duplicated().any():
            raise ValueError("Duplicate wallet identifiers found.")

        result = alerts.copy()

        for column in ["risk_score", "confidence_score"]:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            )

            if result[column].isna().any():
                raise ValueError(f"{column} contains invalid values.")

            if not result[column].between(0, 100).all():
                raise ValueError(f"{column} must be between 0 and 100.")

        valid_levels = {"Low", "Medium", "High", "Critical"}

        if not result["risk_level"].isin(valid_levels).all():
            raise ValueError("Invalid risk level found.")

        result["priority_score"] = (
            result["risk_score"] * 0.7
            + result["confidence_score"] * 0.3
        ).round(2)

        result["priority"] = result["priority_score"].apply(
            self._get_priority
        )

        result = result.sort_values(
            by=["priority_score", "risk_score"],
            ascending=False
        ).reset_index(drop=True)

        result["alert_rank"] = range(1, len(result) + 1)

        return result

    @staticmethod
    def _get_priority(score):
        if score >= 85:
            return "Critical"

        if score >= 70:
            return "High"

        if score >= 45:
            return "Medium"

        return "Low"