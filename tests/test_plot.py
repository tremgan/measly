from conftest import analysed


def test_plot_labels_each_model(data):
    import matplotlib

    matplotlib.use("Agg")
    from measly import plot

    result = analysed(data)
    ax = plot(result, factor=3.0)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(labels) == len(result.models)
    assert all(any(name in label for label in labels) for name in result.models)
    assert ax.get_xlabel() == "training examples"
