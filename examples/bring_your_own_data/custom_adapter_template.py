"""Template: custom :class:`DatasetAdapter` for [YOUR BANK NAME].

Copy this file, fill in the TODOs, and reference it from your config
YAML by dotted path:

    dataset: examples.bring_your_own_data.custom_adapter_template:MyBankAdapter

The training pipeline will import the class, instantiate it with no
arguments, and call :meth:`load_with_protected` with ``data_path`` from
your config.
"""
from __future__ import annotations

import pandas as pd

from workflow.training.datasets import DatasetAdapter, DatasetMetadata


class MyBankAdapter(DatasetAdapter):
    """Replace this docstring with a one-line description of your portfolio."""

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="my_bank_portfolio",     # TODO: short identifier (used in model filename)
            region="IN",                   # TODO: ISO country code (US, IN, EU, ...)
            target_column="default_flag",  # TODO: name of your binary target column
            feature_columns=[
                # TODO: list every column the model should see as input.
                # "loan_amount",
                # "term_months",
                # "interest_rate",
                # "annual_income",
                # "credit_score",
                # "dti_ratio",
                # ...
            ],
            categorical_columns=[
                # TODO: subset of feature_columns that are categorical.
                # "loan_purpose",
                # "employment_status",
            ],
            protected_columns={
                # TODO: columns reserved for fairness EVALUATION ONLY —
                # NOT model inputs. Map column name -> human description.
                # "applicant_gender": "Self-reported gender",
                # "applicant_age_band": "Age bucket: 18-25 / 26-45 / 46-60 / 60+",
                # "applicant_caste": "(India) Constitution Art 15 protected category",
            },
            positive_label=1,              # 1 = "default", "denied", "bad outcome"
            description=(
                # TODO: describe your dataset
                "Internal loan portfolio — replace this description"
            ),
        )

    def load(self, path: str) -> pd.DataFrame:
        """Load and clean the data, returning a DataFrame ready for training.

        The DataFrame must contain every column listed in ``metadata()`` —
        the training pipeline validates this and raises if anything is
        missing.
        """
        df = pd.read_csv(path)

        # TODO: drop / filter rows with missing critical fields
        # df = df.dropna(subset=["loan_amount", "annual_income"])

        # TODO: cap right-tail outliers (XGBoost is robust to outliers but
        # logistic regression / MLP are not — capping is cheap insurance)
        # for col in ("loan_amount", "annual_income"):
        #     cap = df[col].quantile(0.99)
        #     df[col] = df[col].clip(upper=cap)

        # TODO: derive any engineered features (e.g. age bands for
        # fairness evaluation that don't leak into model input)
        # df["applicant_age_band"] = pd.cut(
        #     df["age"], bins=[0, 25, 45, 60, 200],
        #     labels=["18-25", "26-45", "46-60", "60+"],
        # ).astype(str)

        return df

    def download(self, dest_dir: str) -> str:
        """Internal datasets are not downloadable.

        Either raise NotImplementedError (force the user to provide
        ``data_path`` manually) or implement a fetch from your bank's
        internal data lake / S3 bucket / database.
        """
        raise NotImplementedError(
            "Internal datasets are not downloadable. Place your data file "
            "manually and set 'data_path' in the YAML config."
        )
