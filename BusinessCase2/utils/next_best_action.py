import numpy as np
import pandas as pd

# Lifecycle weights — financially motivated
W_RISK  = 0.50
W_EDU   = 0.20
W_AGE   = 0.20
W_WLTH  = 0.10

def calculate_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a personalised risk target combining four client dimensions:
    RiskPropensity, FinancialEducation, Age, and Wealth.
    
    Returns the dataframe with a new 'RiskTarget' column clipped to [0, 1].
    """
    df = df.copy()
    # Normalise log(Wealth) to [0, 1] for the scoring formula
    wealth_log  = np.log1p(df['Wealth'])
    wealth_norm = (wealth_log - wealth_log.min()) / (wealth_log.max() - wealth_log.min())

    df['RiskTarget'] = np.clip(
        W_RISK * df['RiskPropensity']
        + W_EDU  * df['FinancialEducation']
        - W_AGE  * (df['Age'] / 100)
        + W_WLTH * wealth_norm,
        0.0, 1.0
    )
    return df

def baseline_match(client_risk, product_risks, product_ids, product_type_mask):
    """
    Recommend the product of the right type with highest risk < client_risk.
    Returns (product_id=0, risk=0.0) if no suitable product exists.
    """
    mask = product_type_mask & (product_risks < client_risk)
    candidates = product_risks[mask]
    if len(candidates) == 0:
        return 0, 0.0
    best_risk = candidates.max()
    best_id   = product_ids[product_type_mask & (product_risks == best_risk)][0]
    return int(best_id), float(best_risk)

def personalized_match(risk_target, client_risk_propensity, product_risks, product_ids, product_type_mask):
    """
    Recommend the product of the right type with risk closest to risk_target,
    subject to a hard MiFID constraint: product risk <= client RiskPropensity.
    Returns (0, 0.0) if no compliant product exists.
    """
    # Hard MiFID constraint — never exceed client's declared risk tolerance
    mifid_mask = product_type_mask & (product_risks <= client_risk_propensity)
    candidates_risks = product_risks[mifid_mask]
    candidates_ids   = product_ids[mifid_mask]
    if len(candidates_risks) == 0:
        return 0, 0.0
    # Among compliant products, pick the one closest to RiskTarget
    closest_idx = np.argmin(np.abs(candidates_risks - risk_target))
    return int(candidates_ids[closest_idx]), float(candidates_risks[closest_idx])

def evaluate_baseline_approach(results, type_mask, df, prod_risks, prod_ids):
    """
    Evaluates the basic risk ceiling matching strategy across targets.
    Returns a dataframe of matches.
    """
    baseline_rows = []
    for target, r in results.items():
        y_pred = pd.Series(r['y_test_pred'], index=pd.Series(r['y_test_true']).index)
        mask   = type_mask[target]
        positive_idx = y_pred[y_pred == 1].index

        for idx in positive_idx:
            client_risk = float(df.loc[idx, 'RiskPropensity'])
            prod_id, prod_risk = baseline_match(client_risk, prod_risks, prod_ids, mask)
            baseline_rows.append({
                'ClientIdx':       idx,
                'Target':          target,
                'ClientRisk':      client_risk,
                'RecommendedProd': prod_id,
                'ProductRisk':     prod_risk,
                'Matched':         prod_id > 0,
                'Approach':        'Baseline',
            })
    return pd.DataFrame(baseline_rows)

def evaluate_personalized_approach(results, type_mask, df, prod_risks, prod_ids):
    """
    Evaluates the matched recommendations using the predefined 'RiskTarget' property.
    Returns a dataframe of personalized recommendations.
    """
    personalized_rows = []
    for target, r in results.items():
        y_pred = pd.Series(r['y_test_pred'], index=pd.Series(r['y_test_true']).index)
        mask   = type_mask[target]
        positive_idx = y_pred[y_pred == 1].index

        for idx in positive_idx:
            risk_target  = float(df.loc[idx, 'RiskTarget'])
            client_risk  = float(df.loc[idx, 'RiskPropensity'])
            prod_id, prod_risk = personalized_match(
                risk_target, client_risk, prod_risks, prod_ids, mask
            )
            personalized_rows.append({
                'ClientIdx':       idx,
                'Target':          target,
                'ClientRisk':      client_risk,
                'RiskTarget':      risk_target,
                'RecommendedProd': prod_id,
                'ProductRisk':     prod_risk,
                'Gap':             abs(prod_risk - risk_target),
                'Matched':         prod_id > 0,
                'Approach':        'Personalized',
            })
    return pd.DataFrame(personalized_rows)

def evaluate_confidence_approach(results, type_mask, df, prod_risks, prod_ids):
    """
    Integrates xgboost predicted probabilities to decide *whether* to recommend at all,
    using predefined thresholds to group clients by confidence tier.
    """
    confidence_rows = []
    for target, r in results.items():
        y_pred = pd.Series(r['y_test_pred'], index=pd.Series(r['y_test_true']).index)
        y_proba = pd.Series(r['y_test_proba'], index=pd.Series(r['y_test_true']).index)
        mask   = type_mask[target]
        positive_idx = y_pred[y_pred == 1].index

        for idx in positive_idx:
            prob = float(y_proba[idx])
            risk_target  = float(df.loc[idx, 'RiskTarget'])
            client_risk  = float(df.loc[idx, 'RiskPropensity'])

            if prob > 0.75:
                tier = 'High'
            elif prob >= 0.40:
                tier = 'Medium'
            else:
                tier = 'Low'

            if tier != 'Low':
                prod_id, prod_risk = personalized_match(
                    risk_target, client_risk, prod_risks, prod_ids, mask
                )
            else:
                prod_id, prod_risk = 0, 0.0

            confidence_rows.append({
                'ClientIdx':       idx,
                'Target':          target,
                'ClientRisk':      client_risk,
                'RiskTarget':      risk_target,
                'Proba':           prob,
                'ConfidenceTier':  tier,
                'RecommendedProd': prod_id,
                'ProductRisk':     prod_risk,
                'Matched':         prod_id > 0,
            })
    return pd.DataFrame(confidence_rows)
