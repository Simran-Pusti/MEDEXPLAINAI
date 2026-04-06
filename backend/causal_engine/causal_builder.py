import networkx as nx
from dowhy import CausalModel


class CausalBuilder:

    def estimate_causal_effect(self, df, treatment, outcome):

        model = CausalModel(data=df, treatment=treatment, outcome=outcome)

        identified = model.identify_effect()

        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression"
        )

        return estimate.value


    def build_weighted_graph(self, df, target, patient_row=None):

        G = nx.DiGraph()
        weights = {}

        for feature in df.columns:

            if feature == target:
                continue

            try:
                effect = self.estimate_causal_effect(df, feature, target)

                weight = abs(effect)

                #  dynamic adjustment
                if patient_row is not None:
                    weight *= (1 + abs(float(patient_row[feature])))

                weight = round(weight * 100, 2)

                weights[feature] = weight

                G.add_edge(feature, target, weight=weight)

            except:
                pass

        return G, dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))