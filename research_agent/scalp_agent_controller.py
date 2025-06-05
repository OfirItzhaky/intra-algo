import pandas as pd

class ScalpAgentController:
    def __init__(self, max_risk=None, csv_file=None, chart_file=None, session_notes=None):
        self.max_risk = max_risk
        self.csv_file = csv_file
        self.chart_file = chart_file
        self.session_notes = session_notes
        # Add any additional state here as needed

    def run_agent(self):
        """
        Stub for running the scalping agent logic. To be implemented.
        """
        # TODO: Implement agent logic
        return {
            "status": "not_implemented",
            "message": "Scalping agent logic not implemented yet."
        }

    def validate_csv_against_requirements(self, df: pd.DataFrame, requirements: dict):
        """
        Validate the uploaded CSV DataFrame against the requirements dictionary from ScalpAgentSession.
        Returns a dict with is_valid, errors, and validated_df (trimmed to required bar count if valid).
        """
        errors = []
        required_columns = set()
        # 1. Check for indicator columns
        for indicator in requirements.get('detected_indicators', []):
            name = indicator.get('name', '').upper()
            params = indicator.get('parameters', '')
            if name == 'EMA':
                # Expect column like EMA_10_Close for EMA(10)
                param = params.split(',')[0].strip()
                col = f"EMA_{param}_Close"
                if col not in df.columns:
                    errors.append(f"Missing column: {col} (for EMA {param})")
                required_columns.add(col)
            elif name == 'MACD':
                # Expect at least MACD and MACD_Signal
                for col in ["MACD", "MACD_Signal"]:
                    if col not in df.columns:
                        errors.append(f"Missing column: {col} (for MACD)")
                    required_columns.add(col)
            elif name == 'RSI':
                # Expect column like RSI_14 for RSI(14)
                param = params.split(',')[0].strip()
                col = f"RSI_{param}"
                if col not in df.columns:
                    errors.append(f"Missing column: {col} (for RSI {param})")
                required_columns.add(col)
            # TODO: Add more indicator types as needed
        # 2. Special requests
        for req in requirements.get('special_requests', []):
            req_upper = req.upper()
            if 'VWAP' in req_upper:
                if 'VWAP' not in df.columns:
                    errors.append("Missing column: VWAP (for VWAP special request)")
                required_columns.add('VWAP')
            if 'VOLUME AVERAGE' in req_upper or 'AVG VOLUME' in req_upper:
                # Accept either 'Volume_Avg' or 'Avg_Volume'
                if not any(col in df.columns for col in ['Volume_Avg', 'Avg_Volume']):
                    errors.append("Missing column: Volume_Avg or Avg_Volume (for volume average special request)")
                required_columns.update(['Volume_Avg', 'Avg_Volume'])
        # 3. Row count
        required_bar_count = requirements.get('required_bar_count')
        try:
            required_bar_count = int(required_bar_count)
        except (TypeError, ValueError):
            required_bar_count = None
        if required_bar_count is not None and len(df) < required_bar_count:
            errors.append(f"Not enough rows: required {required_bar_count}, found {len(df)}")
        # 4. Optional: interval check (stub)
        # TODO: Implement interval check if interval info is available in DataFrame or filename
        # 5. Prepare validated_df
        validated_df = None
        if not errors:
            if required_bar_count is not None:
                validated_df = df.tail(required_bar_count)
            else:
                validated_df = df.copy()
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'validated_df': validated_df
        }

    def generate_trade_idea(self, df, max_risk_per_trade=None):
        """
        Generate a mock trade idea using the validated DataFrame.
        If max_risk_per_trade is provided, calculate position size.
        """
        # Assume a fixed stop loss per contract (e.g., $50 risk per MES contract)
        stop_loss_per_contract = 50  # dollars per contract (example)
        position_size = None
        position_size_warning = None
        if max_risk_per_trade is not None:
            try:
                max_risk_per_trade = float(max_risk_per_trade)
                position_size = int(max_risk_per_trade // stop_loss_per_contract)
                if position_size < 1:
                    position_size_warning = "Max risk per trade is too low for minimum position size (risk < stop loss per contract)."
            except Exception:
                position_size_warning = "Invalid max_risk_per_trade value."
        trade_idea = {
            "direction": "Long",
            "entry_rule": "Price crosses above EMA(10)",
            "entry_type": "Limit",
            "stop_loss": "Below recent swing low (e.g., 0.75% risk)",
            "take_profit": ["1.5x risk", "trail stop after 1R"],
            "strategy_label": "EMA Reversal V1",
            "explanation": "The price is bouncing off EMA(10) with strong volume. Expecting trend continuation."
        }
        if max_risk_per_trade is not None:
            trade_idea["position_size"] = position_size if position_size and position_size > 0 else 0
            if position_size_warning:
                trade_idea["explanation"] += f" WARNING: {position_size_warning}"
        return {
            "is_successful": True,
            "trade_idea": trade_idea
        } 