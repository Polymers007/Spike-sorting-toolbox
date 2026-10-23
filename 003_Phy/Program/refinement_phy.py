from __future__ import annotations

import argparse
from pathlib import Path

from phylib.io.model import load_model
from phylib.utils import connect
from phy.apps.template.gui import TemplateController
from phy.gui import create_app, run_app
from PyQt5.QtCore import QTimer

ACTIVE_FILTER = ""
ORDER_COLUMN = "refinement_order"
MAX_CORRELOGRAM_SPIKES = 50_000
LEAN_VIEW_COUNT = {
    "ClusterView": 1,
    "SimilarityView": 1,
    "WaveformView": 1,
    "AmplitudeView": 1,
    "CorrelogramView": 1,
}

def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Phy final refinement with all clusters visible.")
    parser.add_argument("params", type=Path)
    args = parser.parse_args()
    params = args.params.resolve(); folder = params.parent
    model = load_model(params)
    create_app()
    controller = TemplateController(model=model, dir_path=folder)

    controller.n_spikes_correlograms = MAX_CORRELOGRAM_SPIKES
    supervisor = controller.supervisor

    def refinement_order(cluster_id: int) -> int:
        try:
            channel = int(controller.get_best_channel(cluster_id))
        except Exception:
            channel = 1_000_000
        return channel * 1_000_000 + int(cluster_id)

    supervisor.cluster_metrics[ORDER_COLUMN] = refinement_order
    if ORDER_COLUMN not in supervisor.columns:
        supervisor.columns.append(ORDER_COLUMN)
    gui = controller.create_gui(view_count=LEAN_VIEW_COUNT)

    def configure_views(*_args, **_kwargs):
        cluster_view = supervisor.cluster_view
        similarity_view = supervisor.similarity_view

        if cluster_view._filter_text != ACTIVE_FILTER:
            cluster_view.filter(ACTIVE_FILTER)
        if similarity_view._filter_text != ACTIVE_FILTER:
            similarity_view.filter(ACTIVE_FILTER)
        if cluster_view._current_sort != (ORDER_COLUMN, "asc"):
            cluster_view.sort_by(ORDER_COLUMN, "asc")
        order_index = cluster_view.columns.index(ORDER_COLUMN)
        cluster_view.table_view.setColumnHidden(order_index, True)
        if ORDER_COLUMN in similarity_view.columns:
            similarity_view.table_view.setColumnHidden(similarity_view.columns.index(ORDER_COLUMN), True)
        cluster_view.filter_edit.setReadOnly(False)
        similarity_view.filter_edit.setReadOnly(False)

    def refresh_order_and_selection(selected_before: list[int], added: list[int]):
        cluster_view = supervisor.cluster_view
        patches = []
        for cluster_id in supervisor.clustering.cluster_ids:
            cluster_id = int(cluster_id)
            value = refinement_order(cluster_id)
            row = cluster_view._model.row_by_id(cluster_id)
            if row is not None and row.get(ORDER_COLUMN) != value:
                patches.append({"id": cluster_id, ORDER_COLUMN: value})
        if patches:
            cluster_view.change(patches)
        configure_views()

        visible = set(cluster_view._visible_ids())

        desired = [cluster_id for cluster_id in selected_before if cluster_id in visible]
        if not desired:
            desired = [cluster_id for cluster_id in added if cluster_id in visible]
        if not desired:
            desired = [cluster_id for cluster_id in cluster_view.get_selected_ids() if cluster_id in visible]
        if desired:
            if cluster_view.get_selected_ids() != desired:
                cluster_view.select(desired)
            cluster_view.scroll_to(desired[-1])

    @connect(sender=supervisor)
    def on_cluster(sender, up):
        selected_before = [int(value) for value in supervisor.cluster_view.get_selected_ids()]
        added = [int(value) for value in (getattr(up, "added", None) or [])]
        QTimer.singleShot(75, lambda: refresh_order_and_selection(selected_before, added))

    @connect(sender=supervisor.cluster_view)
    def on_ready(sender):
        QTimer.singleShot(0, configure_views)

    @connect(sender=supervisor.similarity_view)
    def on_ready(sender):
        QTimer.singleShot(0, configure_views)

    QTimer.singleShot(0, configure_views)
    QTimer.singleShot(350, configure_views)
    gui.show(); run_app(); gui.close(); controller.model.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
