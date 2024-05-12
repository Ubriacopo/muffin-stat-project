import pandas
import plotly.express as px
import plotly.graph_objs as go


def make_validation_metric_graph(data: pandas.DataFrame, metric: str, is_with_tuner_iteration: bool):
    """
    todo explain why
    :param data:
    :param metric:
    :param is_with_tuner_iteration:
    :return:
    """
    base_figure = px.line(data, x="epoch", y=[str.lower(metric)], markers=True,
                          color="tuner_iteration" if is_with_tuner_iteration else None)

    validation_figure = px.line(data, x="epoch", y=[f"val_{str.lower(metric)}"], markers=True,
                                color="tuner_iteration" if is_with_tuner_iteration else None)
    validation_figure.update_traces(patch={"line": {"dash": 'dot'}})

    return_figure = go.Figure(data=base_figure.data + validation_figure.data)
    return_figure.update_layout(template="plotly_white", xaxis_title="Epoch")

    return return_figure


# Accuracy
def make_loss_graphs(data: pandas.DataFrame, is_with_tuner_iteration: bool = True) -> go.Figure:
    figure = make_validation_metric_graph(data, "loss", is_with_tuner_iteration)
    return figure.update_layout(yaxis_title="Loss")


# Loss
def make_loss_accuracy_graphs(data: pandas.DataFrame, is_with_tuner_iteration: bool = True) -> go.Figure:
    figure = make_validation_metric_graph(data, "accuracy", is_with_tuner_iteration)
    return figure.update_layout(yaxis_title="Accuracy")


def add_tuner_iteration_to_data(data: pandas.DataFrame):
    """
    todo doc
    :param data:
    :return:
    """
    data['tuner_iteration'] = 0

    current_iteration = -1
    for index, row in enumerate(data.itertuples()):
        if data.at[index, 'epoch'] == 0:
            current_iteration += 1
        data.at[index, 'tuner_iteration'] = current_iteration
