import pandas as pd


class DatasetLoader:

    def load_dataset(self, file):

        if file.name.endswith(".csv"):

            try:
                # Try normal comma separator
                df = pd.read_csv(file)

                # If only one column detected, retry with ;
                if df.shape[1] == 1:
                    file.seek(0)
                    df = pd.read_csv(file, sep=";")

            except:
                file.seek(0)
                df = pd.read_csv(file, sep=";")

        elif file.name.endswith(".xlsx"):

            df = pd.read_excel(file)

        else:
            raise Exception("Unsupported file format")

        # ---------- FIX: CLEAN NUMERIC VALUES ----------
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    # Remove brackets like [4.99E-1]
                    df[col] = df[col].astype(str).str.replace("[", "", regex=False)
                    df[col] = df[col].str.replace("]", "", regex=False)

                    # Convert to numeric where possible
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except:
                    pass
        # -----------------------------------------------

        return df