import pandas as pd
import numpy as np

import itertools
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

def is_proba(s:pd.Series):
    return (
        pd.api.types.is_numeric_dtype(s)
        and not pd.api.types.is_bool_dtype(s)
        and s.between(0, 1).all()
    )


class ClassificationQCError(ValueError):
    def __repr__(self): return str(self)


def qcplot_dataframes(fact, pred, kind='all', rounding=2, pxfigsize=(600,400), template='plotly_dark'):

    if not fact.columns.symmetric_difference(pred.columns).empty:
        raise ClassificationQCError(
            'columns must be the same, as we are '
            'comparing all fact levels to pred levels') from None

    if (~fact.apply(lambda s: s.between(0,1))).any().any() or (~pred.apply(lambda s: s.between(0,1))).any().any():
        raise ClassificationQCError(
            'individual class columns can only '
            'be in range 0:1 (in a one-hot fashion)') from None

    class_fractions = fact.sum().pipe(lambda s: s/s.sum())

    if kind == 'all' or 'frac' in kind:
        (
            px.bar(
                fact.sum().sort_values().rename('counts'),
                title=f'Class counts',
                width=pxfigsize[0],
                height=pxfigsize[1],
                hover_data={'fraction':class_fractions.round(3)}
            )
            .add_hline(fact.sum().mean(), annotation={'text':'balanced'})
            .update_layout(showlegend=False, margin={'t':80})
            .show()
        )
    if kind=='all' or 'roc' in kind:

        rocs = go.Figure(
            layout={
                'width':pxfigsize[0],
                'height':pxfigsize[1],
                'template':template,
                'xaxis':{'title':'FPR'},
                'yaxis':{'title':'TPR'},
            })

        for name_of_class, color in zip(
            fact.columns,
            itertools.cycle(px.colors.qualitative.Safe)
        ):
            fpr, tpr, thr = roc_curve(fact[name_of_class], pred[name_of_class].round(rounding))
            rocs.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{name_of_class} (ROC-AUC = {roc_auc_score(fact[name_of_class], pred[name_of_class]):.3f})',
                    customdata = thr,
                    hovertemplate="<br>".join([
                        '<b>%{fullData.name}</b><br>',
                        'fpr: %{x:.2f}',
                        'tpr: %{y:.2f}',
                        'threshold: %{customdata:.2f}',
                        '<extra></extra>']),
                    line_color=color
            ))

        rocs.add_trace(
            go.Scatter(
                x=[0,1],y=[0,1],
                mode='lines',
                showlegend=False,
                line={'dash':'dash', 'color':'grey'},
        ))

        rocs.show()


    if kind=='all' or 'pr' in kind:

        prs = go.Figure(
            layout={
                'width':pxfigsize[0],
                'height':pxfigsize[1],
                'template':template,
                'xaxis':{'title':'Recall'},
                'yaxis':{'title':'Precision'},
            })

        for name_of_class, color in zip(
            fact.columns,
            itertools.cycle(px.colors.qualitative.Safe)
        ):

            precision, recall, thr = precision_recall_curve(fact[name_of_class], pred[name_of_class].round(rounding))
            prs.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    name=f'{name_of_class} (AP = {average_precision_score(fact[name_of_class], pred[name_of_class]):.3f})',
                    customdata = thr,
                    hovertemplate="<br>".join([
                        '<b>%{fullData.name}</b><br>',
                        'recall: %{x:.2f}',
                        'precision: %{y:.2f}',
                        'threshold: %{customdata:.2f}',
                        f'<br><i>no-skill precision: {class_fractions[name_of_class]:.2f}</i>',
                        '<extra></extra>']),
                    line_color=color
            ))

            prs.add_trace(
                go.Scatter(
                    x=[0,0,1],
                    y=[1, class_fractions[name_of_class], class_fractions[name_of_class]],
                    mode='lines',
                    showlegend=False,
                    name=f'No skill baseline for category "{name_of_class}"',
                    hovertemplate=f'<i>no-skill precision: {class_fractions[name_of_class]:.2f}</i>',
                    line={'dash':'dash', 'color': color},
                    opacity=0.4
            ))
        prs.show()


def qcplot(fact, pred, kind='all', rounding=2, pxfigsize=(800,400), template='plotly_dark'):

        if isinstance(fact, pd.Series) and isinstance(pred, pd.Series):

            if not (is_proba(fact) and is_proba(pred)):

                raise ClassificationQCError(
                    'Passing two series implies a simplified binary classification, '
                    'with necessarily numeric values. Series must be between(0,1)') from None

            return qcplot(
                fact,
                pd.DataFrame({
                    1: pred,
                    0: 1-pred
                }),
                kind=kind,
                rounding=rounding,
                pxfigsize=pxfigsize,
                template=template
            )

        if isinstance(fact, pd.Series) and isinstance(pred, pd.DataFrame):

            fact_levels = pd.Index(fact.unique())
            missing_in_fact = pred.columns.difference(fact_levels)
            missing_in_pred = fact_levels.difference(pred.columns)
            if not missing_in_fact.empty:
                print(f'qcplot: `fact` is missing levels {missing_in_fact.to_list()}')
            if not missing_in_pred.empty:
                print(f'qcplot: `pred` is missing level columns {missing_in_pred.to_list()}')

            levels = pred.columns.intersection(fact_levels)
            fact_onehot = label_binarize(fact, classes=levels)

            if len(levels) == 2:
                fact_onehot = np.c_[1 - fact_onehot, fact_onehot] # order matters, last level of the two is the positive case

            return qcplot(
                pd.DataFrame(fact_onehot, columns=levels),
                pred.reindex(columns=levels),
                kind=kind,
                rounding=rounding,
                pxfigsize=pxfigsize,
                template=template
            )

        if isinstance(fact, pd.DataFrame) and isinstance(pred, pd.DataFrame):
            return qcplot_dataframes(
                fact,
                pred,
                kind=kind,
                rounding=rounding,
                pxfigsize=pxfigsize,
                template=template
            )

        raise ClassificationQCError(
            f'`fact` (of type "{type(fact)}") and `pred` (of type "{type(pred)}")'
            'dont match any of the possible combinations, and thus, outcomes:\n'
            '#1. DataFrame, DataFrame       => column-wise comparison\n'
            '#2. Series,    DataFrame       => one-hot encoding and then #1\n'
            '#3. Series,    Series          => conversion for binary classification and then #1'
        ) from None




class ThresholdClassifier:

    def __init__(self, thresholds = None):
        self.thresholds = thresholds if thresholds else {}

    def fit(self, fact, probas, features_of_ordered_importance:'dict[str, float] | None' = None, overwrite=False):
        if features_of_ordered_importance is not None:
            for col, beta in features_of_ordered_importance.items():
                if overwrite or not col in self.thresholds:
                    precision, recall, thresholds = precision_recall_curve(fact[col], probas[col])
                    fbeta = ((1+beta**2) * precision * recall) / (beta ** 2 * precision + recall)
                    self.thresholds[col] = thresholds[np.argmax(fbeta)]
        return self

    def predict(self, probas):
        res = pd.Series(index=probas.index, dtype=float)
        for col, threshold in self.thresholds.items():
            res[(probas[col] > threshold) & res.isna()] = col
        return res.fillna(probas.idxmax(axis=1))
