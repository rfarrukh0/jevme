"""Workflow mining/replay tests are deterministic and platform-independent."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from jevme import config, ui_memory
from jevme.workflows import (
    EPISODE_IDLE_S,
    MAX_DISMISSED,
    MAX_WORKFLOWS,
    MIN_OCCURRENCES,
    PENDING_TTL_S,
    SCHEMA_VERSION,
    Workflow,
    WorkflowCandidate,
    WorkflowPersistenceError,
    WorkflowRunner,
    WorkflowStore,
    WorkflowUnit,
    eligible_recipe,
    eligible_tool,
    recipe_unit,
    register_tools,
    tool_unit,
)


def unit(app: str) -> WorkflowUnit:
    result = tool_unit("open_app", {"app": app})
    assert result is not None
    return result


def repeat(store: WorkflowStore, units: list[WorkflowUnit], *, start: float) -> float:
    last = start
    for occurrence in range(MIN_OCCURRENCES):
        base = start + occurrence * (EPISODE_IDLE_S + len(units) + 2)
        for offset, item in enumerate(units):
            last = base + offset
            assert store.record(item, last) is None
    return last


def finish(store: WorkflowStore, last: float):
    return store.finalize_candidate(last + EPISODE_IDLE_S + 1)


def workflow_json(ident: str, name: str, units: list[WorkflowUnit]) -> dict:
    return {"id": ident, "name": name, "units": [item.to_dict() for item in units],
            "created": time.time(), "hits": 0}


def test_candidate_waits_for_idle_and_chooses_longest_across_restart(tmp_path):
    path = tmp_path / "workflows.json"
    start = time.time()
    routine = [unit("Slack"), unit("Jira"), unit("Calendar")]
    store = WorkflowStore(path)
    last = repeat(store, routine, start=start)
    assert store.pending is None  # no A-B prefix is surfaced while A-B-C can still continue

    store = WorkflowStore(path)
    found = finish(store, last)
    assert found is not None
    assert found.units == tuple(routine)
    assert found.occurrences == MIN_OCCURRENCES
    loaded = WorkflowStore(path)
    assert loaded.pending is not None and loaded.pending.signature == found.signature


def test_episode_boundaries_and_overlaps_do_not_create_false_candidates(tmp_path):
    store = WorkflowStore(tmp_path / "w.json")
    a, b = unit("Slack"), unit("Jira")
    start = time.time()
    for occurrence in range(MIN_OCCURRENCES):
        base = start + occurrence * EPISODE_IDLE_S * 3
        store.record(a, base)
        store.record(b, base + EPISODE_IDLE_S + 1)
    assert finish(store, base + EPISODE_IDLE_S + 1) is None

    store = WorkflowStore(tmp_path / "overlap.json")
    for offset in range(4):
        store.record(a, start + offset)
    assert store._non_overlapping_count((a, a, a)) == 1
    assert finish(store, start + 3) is None


def test_pending_expires_and_new_unrelated_routine_replaces_it(tmp_path):
    store = WorkflowStore(tmp_path / "w.json")
    start = time.time()
    last = repeat(store, [unit("Slack"), unit("Jira")], start=start)
    first = finish(store, last)
    assert first is not None

    later = last + EPISODE_IDLE_S + 10
    last = repeat(store, [unit("Notes"), unit("Calendar")], start=later)
    second = finish(store, last)
    assert second is not None and second.signature != first.signature
    assert store.pending is second

    assert store.finalize_candidate(second.detected + PENDING_TTL_S + 1) is None
    assert store.pending is None


def test_privacy_and_central_safety_filters():
    assert eligible_tool("open_app", {"app": "Slack"})
    assert not eligible_tool("read_screen", {})
    assert not eligible_tool("press_key", {"key": "enter"})
    assert not eligible_tool("send_draft", {})
    assert not eligible_tool("open_app", {"app": "Delete Account"})
    assert not eligible_tool("open_site", {"site": "example.com/private?q=screen text"})

    safe = [ui_memory.RecipeStep("open_app", arg="Notes"),
            ui_memory.RecipeStep("click", role="AXButton", label="New Note")]
    assert eligible_recipe("open notes and create a note", safe)
    for unsafe in ("delete old note", "move it to trash", "send an email", "submit form",
                   "checkout", "place order", "pay bill", "unsubscribe"):
        assert not eligible_recipe(unsafe, safe)
    assert not eligible_recipe("open notes", safe + [ui_memory.RecipeStep("key", arg="return")])
    assert not eligible_recipe("open notes", safe + [ui_memory.RecipeStep("type", arg="private words")])


def test_recipe_persistence_is_an_opaque_reference_only(tmp_path):
    goal = "open notes and create a note"
    steps = [ui_memory.RecipeStep("open_app", arg="Notes"),
             ui_memory.RecipeStep("click", role="AXButton", label="New Note")]
    recipe = recipe_unit(goal, "Finder", steps)
    assert recipe is not None
    payload = json.dumps(recipe.to_dict())
    assert set(recipe.to_dict()) == {"kind", "recipe_key"}
    for private in (goal, "Finder", "New Note", "AXButton"):
        assert private not in payload

    store = WorkflowStore(tmp_path / "w.json")
    store.record(recipe, time.time())
    serialized = json.dumps(store._data())
    for private in (goal, "Finder", "New Note", "AXButton"):
        assert private not in serialized


def test_save_collisions_duplicates_and_persistence_failure(tmp_path, monkeypatch):
    store = WorkflowStore(tmp_path / "w.json")
    start = time.time()
    last = repeat(store, [unit("Slack"), unit("Jira")], start=start)
    finish(store, last)
    saved, message = store.save_pending("morning standup")
    assert saved is not None and message == "Saved workflow: morning standup"

    last = repeat(store, [unit("Notes"), unit("Calendar")], start=last + EPISODE_IDLE_S + 10)
    finish(store, last)
    other, message = store.save_pending("Morning Standup")
    assert other is None and "already exists" in message
    assert store.pending is not None

    monkeypatch.setattr(store, "save", lambda: False)
    pending = store.pending
    with pytest.raises(WorkflowPersistenceError):
        store.save_pending("evening")
    assert store.pending is pending
    assert store.by_name("evening") is None


@pytest.mark.parametrize("contents", [
    "{not json",
    json.dumps({"version": 999, "workflows": []}),
    json.dumps({"version": SCHEMA_VERSION, "workflows": "not-a-list"}),
])
def test_incompatible_files_are_preserved_and_never_overwritten(tmp_path, contents):
    path = tmp_path / "w.json"
    path.write_text(contents, encoding="utf-8")
    store = WorkflowStore(path)
    store.record(unit("Slack"), time.time())
    assert not store.save()
    assert path.read_text(encoding="utf-8") == contents


def test_valid_entries_recover_independently_and_bounds_are_enforced(tmp_path):
    path = tmp_path / "w.json"
    valid = workflow_json("valid", "Valid Routine", [unit("Slack"), unit("Jira")])
    workflows = [{"id": "bad id!", "name": "x", "units": []}, valid]
    workflows.extend(workflow_json(f"wf-{i}", f"Routine {i}", [unit("Notes"), unit("Calendar")])
                     for i in range(MAX_WORKFLOWS + 10))
    data = {"version": SCHEMA_VERSION, "workflows": workflows,
            "history": [{"bad": True}, {"unit": unit("Slack").to_dict(),
                                          "timestamp": time.time(), "episode": 1}],
            "dismissed": {f"sig-{i}": time.time() for i in range(MAX_DISMISSED + 20)}}
    path.write_text(json.dumps(data), encoding="utf-8")
    store = WorkflowStore(path)
    assert store.by_name("Valid Routine") is not None
    assert len(store.workflows) == MAX_WORKFLOWS
    assert len(store.history) == 1
    assert len(store.dismissed) == MAX_DISMISSED

    too_long = workflow_json("too-long", "Too Long", [unit(str(i)) for i in range(7)])
    data["workflows"] = [too_long]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert WorkflowStore(path).workflows == []


def test_runner_revalidates_each_unit_and_stops_on_stale_or_failed_tool(tmp_path):
    store = WorkflowStore(tmp_path / "w.json")
    unsafe = WorkflowUnit("tool", tool_name="read_screen", args={})
    store.workflows.append(Workflow("unsafe", "unsafe", (unit("Slack"), unsafe)))
    calls = []
    runner = WorkflowRunner(store, run_tool=lambda name, args, label, record: calls.append((name, record)) or True,
                            replay_recipe=lambda *args: True, cancelled=lambda: False)
    with pytest.raises(RuntimeError, match="no longer eligible"):
        runner.run("unsafe")
    assert calls == [("open_app", False)]

    store.workflows = [Workflow("stale", "stale", (unit("Slack"), unit("Jira")))]
    runner = WorkflowRunner(store, run_tool=lambda *args, **kwargs: False,
                            replay_recipe=lambda *args: True, cancelled=lambda: False)
    with pytest.raises(RuntimeError, match="step 1"):
        runner.run("stale")
    assert store.workflows[0].hits == 0


def test_workflow_replay_does_not_train_itself_and_disabled_mining_still_runs(tmp_path, monkeypatch):
    store = WorkflowStore(tmp_path / "w.json")
    workflow = Workflow("morning", "morning", (unit("Slack"), unit("Jira")))
    store.workflows.append(workflow)
    calls = []
    monkeypatch.setattr(config, "WORKFLOW_LEARNING", False)
    runner = WorkflowRunner(store,
                            run_tool=lambda name, args, label, record: calls.append(record) or True,
                            replay_recipe=lambda *args: True, cancelled=lambda: False)
    assert runner.run("morning") == "Finished workflow: morning"
    assert calls == [False, False]
    assert store.history == []
    assert workflow.hits == 1


def test_recipe_failure_and_cancellation_stop_without_counting_completion(tmp_path):
    store = WorkflowStore(tmp_path / "w.json")
    recipe = WorkflowUnit("recipe", recipe_key="a" * 64)
    workflow = Workflow("recipe", "recipe", (recipe, unit("Slack")))
    store.workflows.append(workflow)
    runner = WorkflowRunner(store, run_tool=lambda *args, **kwargs: True,
                            replay_recipe=lambda *args: False, cancelled=lambda: False)
    with pytest.raises(RuntimeError, match="step 1"):
        runner.run("recipe")
    assert workflow.hits == 0

    calls = []
    store.workflows = [Workflow("cancel", "cancel", (unit("Slack"), unit("Jira")))]
    runner = WorkflowRunner(store,
                            run_tool=lambda name, args, label, record: calls.append(name) or True,
                            replay_recipe=lambda *args: True, cancelled=lambda: bool(calls))
    with pytest.raises(RuntimeError, match="cancelled"):
        runner.run("cancel")
    assert calls == ["open_app"]
    assert store.workflows[0].hits == 0


def test_startup_reload_and_dynamic_registration(tmp_path):
    path = tmp_path / "w.json"
    data = {"version": SCHEMA_VERSION,
            "workflows": [workflow_json("morning", "Morning Routine", [unit("Slack"), unit("Jira")])],
            "history": [], "pending": None, "dismissed": {}}
    path.write_text(json.dumps(data), encoding="utf-8")
    store = WorkflowStore(path)

    class FakeTool:
        def __init__(self, name, description, examples, run, **kwargs):
            self.name, self.run, self.kwargs = name, run, kwargs

    fake_tools = SimpleNamespace(Tool=FakeTool,
                                 TextArg=lambda *args, **kwargs: (args, kwargs),
                                 EnumArg=lambda *args, **kwargs: (args, kwargs),
                                 TOOLS=[], BY_NAME={})
    learned = []
    fake_vocab = SimpleNamespace(learn=lambda *names: learned.extend(names))
    ran = []
    register_tools(store, lambda name: ran.append(name) or "done",
                   tools_module=fake_tools, vocab_module=fake_vocab)
    assert set(fake_tools.BY_NAME) == {"save_workflow", "run_workflow"}
    assert learned == ["Morning Routine"]
    assert fake_tools.BY_NAME["run_workflow"].run({"workflow": "Morning Routine"}) == "done"
    assert ran == ["Morning Routine"]
