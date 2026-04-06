class DatasetValidator:

    def validate(self, df):

        report = {}

        report["rows"] = df.shape[0]
        report["columns"] = df.shape[1]
        report["missing_values"] = df.isnull().sum().sum()

        report["column_types"] = dict(df.dtypes)

        return report